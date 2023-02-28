import math
import torch
import torch.nn as nn
import torch_redstone as rst
import torch.nn.functional as F
from collections import namedtuple


MethodDesc = namedtuple('MethodDesc', ['loss', 'metrics', 'processors'])


def natural(_):
    return MethodDesc(None, [], [])


def trades_fgsm(args):
    return MethodDesc(TradesTrainLoss(args.beta), [Trades(), CR()], [TradesProcessor(args.eps, 1, 1.25)])


def trades_pgd(args):
    return MethodDesc(TradesTrainLoss(args.beta), [Trades(), CR()], [TradesProcessor(args.eps, 7, 0.25)])


def trades_pi_fgsm(args):
    pitrades = PITradesTrainLoss()
    return MethodDesc(pitrades, [
        Trades(),
        Beta(pitrades),
        CR()
    ], [TradesProcessor(args.eps, 1, 1.25)])


def pi_fgsm(args):
    piracc = PIRobustAccLoss()
    return MethodDesc(piracc, [
        rst.CategoricalAcc().redstone('RAcc', 'adv_logits'),
        nn.CrossEntropyLoss().redstone('RLoss', 'adv_logits'),
        Beta(piracc)
    ], [PostATProcessor(args.eps, 1, 1.25)])


def pi_pgd(args):
    piracc = PIRobustAccLoss()
    return MethodDesc(piracc, [
        rst.CategoricalAcc().redstone('RAcc', 'adv_logits'),
        nn.CrossEntropyLoss().redstone('RLoss', 'adv_logits'),
        Beta(piracc)
    ], [PostATProcessor(args.eps, 7, 0.5)])


def pgd(args):
    return MethodDesc(None, [], [rst.AdvTrainingPGD(nn.CrossEntropyLoss().redstone(), [], args.eps, 0.5, 7)])


class TradesProcessor(rst.Processor):
    def __init__(self, eps, steps, ratio) -> None:
        super().__init__()
        self.criterion_kl = nn.KLDivLoss(reduction='sum')
        self.eps = eps
        self.steps = steps
        self.ratio = ratio

    def post_forward(self, inputs, model: nn.Module, model_return):
        old_training = model.training
        model.eval()
        x_natural = inputs.x
        with torch.no_grad():
            natural_logits = model(x_natural)
        x_adv = x_natural.detach() + self.eps * 2 * (torch.rand_like(x_natural) - 0.5)
        for _ in range(self.steps):
            x_adv.requires_grad_()
            with torch.enable_grad():
                loss_kl = self.gscaler.scale(self.criterion_kl(F.log_softmax(model(x_adv), dim=1),
                                                               F.softmax(natural_logits, dim=1)))
            grad = torch.autograd.grad(loss_kl, [x_adv])[0]
            x_adv = x_adv.detach() + self.eps * self.ratio * torch.sign(grad.detach())
            x_adv = torch.min(torch.max(x_adv, x_natural - self.eps), x_natural + self.eps)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)
        model.train(old_training)
        x_adv = x_adv.detach()
        adv_pred = model(x_adv)
        nat_pred = model_return.logits
        loss_robust = (1. / len(x_natural)) * self.criterion_kl(F.log_softmax(adv_pred, dim=1),
                                                                F.softmax(nat_pred, dim=1))
        model_return.trades = loss_robust
        model_return.cr = (adv_pred.argmax(1) == nat_pred.argmax(1)).float().mean()
        return model_return


class PostATProcessor(rst.Processor):
    
    def __init__(self, eps, steps, ratio) -> None:
        super().__init__()
        self.ce = nn.CrossEntropyLoss()
        self.cacc = rst.CategoricalAcc()
        self.eps = eps
        self.steps = steps
        self.ratio = ratio

    def post_forward(self, inputs, model: nn.Module, model_return):
        old_training = model.training
        model.eval()
        x_natural = inputs.x
        x_adv = x_natural.detach() + self.eps * 2 * (torch.rand_like(x_natural) - 0.5)
        for _ in range(self.steps):
            x_adv.requires_grad_()
            with torch.enable_grad():
                loss_kl = self.gscaler.scale(self.ce(model(x_adv), inputs.y))
            grad = torch.autograd.grad(loss_kl, [x_adv])[0]
            x_adv = x_adv.detach() + self.eps * self.ratio * torch.sign(grad.detach())
            x_adv = torch.min(torch.max(x_adv, x_natural - self.eps), x_natural + self.eps)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)
        model.train(old_training)
        x_adv = x_adv.detach()
        model_return.adv_logits = model(x_adv)
        return model_return


class Trades(rst.Metric):
    def __call__(self, inputs, model_return) -> torch.Tensor:
        return model_return.trades


class CR(rst.Metric):
    def __call__(self, inputs, model_return) -> torch.Tensor:
        return model_return.cr


class TradesTrainLoss(rst.Loss):
    def __init__(self, beta) -> None:
        super().__init__()
        self.beta = beta

    def __call__(self, inputs, model_return, metrics) -> torch.Tensor:
        return metrics.loss + self.beta * metrics.trades


class PIControl(object):
    def __init__(self, init, vmin):
        self.I_k1 = 0.0
        self.W_k1 = init
        self.vmin = vmin

    def _Kp_fun(self, err, scale=1):
        return 1.0 / (1.0 + float(scale) * math.exp(err))

    def __call__(self, target, current, Kp=0.5, Ki=0.0):
        error_k = target - current
        # compute U as the control factor
        Pk = Kp * error_k
        Ik = self.I_k1 + Ki * error_k

        # wind-up for integrator
        if self.W_k1 < self.vmin:
            Ik = self.I_k1
        Wk = Pk + Ik + self.W_k1
        self.W_k1 = Wk
        self.I_k1 = Ik
        
        # clamp min
        if Wk < self.vmin:
            Wk = self.vmin
        
        return Wk, error_k


class PILoss(rst.Loss):
    def __init__(self) -> None:
        super().__init__()
        self.beta = 10.0
        self.target_v = 0.6
        self.beta_min = 0.01
        self.pid = PIControl(self.beta, self.beta_min)
        self.momentum = 0.8
        self.ema = 0.0
        self.emw = 0.0

    def target(self, metrics):
        raise NotImplementedError

    def control(self, metrics):
        return self.target(metrics)

    def __call__(self, inputs, model_return, metrics) -> torch.Tensor:
        trades = self.control(metrics).item()
        self.ema = self.ema * self.momentum + trades * (1 - self.momentum)
        self.emw = self.emw * self.momentum + (1 - self.momentum)
        self.beta, _ = self.pid(self.target_v, self.ema / self.emw)
        return metrics.loss + self.beta * self.target(metrics)


class PITradesTrainLoss(PILoss):
    def target(self, metrics):
        return metrics.trades

    def control(self, metrics):
        return metrics.cr


class PIRobustAccLoss(PILoss):
    def target(self, metrics):
        return metrics.rloss

    def control(self, metrics):
        return metrics.racc


class Beta(rst.Metric):
    name = "β"

    def __init__(self, pid_trades: PILoss) -> None:
        self.pid_trades = pid_trades

    def __call__(self, inputs, model_return) -> torch.Tensor:
        return torch.tensor(self.pid_trades.beta)
