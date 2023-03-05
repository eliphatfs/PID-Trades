import time
import torch
import torch_redstone as rst
from typing import Type, Dict
from argparse import ArgumentParser

import dyntrades

tasks: Dict[str, Type[dyntrades.tasks.ImageClassificationTask]] = {
    'mnist': dyntrades.tasks.MNISTTask,
    'cifar10': dyntrades.tasks.CIFAR10Task,
}
models = {
    'resnet18': dyntrades.networks.resnet18,
    'vgg13': dyntrades.networks.vgg13,
}
methods = {
    'natural': dyntrades.methods.natural,
    'ftrades': dyntrades.methods.trades_fgsm,
    'trades': dyntrades.methods.trades_pgd,
    'piftrades': dyntrades.methods.trades_pi_fgsm,
    'pifgsm': dyntrades.methods.pi_fgsm,
    'pipgd': dyntrades.methods.pi_pgd,
    'pgd': dyntrades.methods.pgd,
    'pitrades': dyntrades.methods.trades_pi_pgd,
    'fgsm': dyntrades.methods.fgsm,
    'cyctrades': dyntrades.methods.trades_cyc_pgd,
}


def main():
    p = ArgumentParser()
    p.add_argument('-t', '--task', choices=sorted(tasks.keys()))
    p.add_argument('-m', '--model', choices=sorted(models.keys()))
    p.add_argument('-e', '--method', choices=sorted(methods.keys()))
    p.add_argument('--beta', type=float, default=2.0)
    p.add_argument('--eps', type=float, default=8/255)
    p.add_argument('--epochs', type=int, default=62)
    p.add_argument('--batch_size', type=int, default=256)
    p.add_argument('--robust_test', action='store_true')
    p.add_argument('--no_cuda', action='store_true')
    p.add_argument('--no_amp', action='store_true')
    p.add_argument('--load_model', default=None)

    args = p.parse_args()

    task = tasks[args.task]()
    model = models[args.model](task.num_classes)
    mthd_desc: dyntrades.methods.MethodDesc = methods[args.method](args)
    if args.robust_test:
        mthd_desc = dyntrades.methods.MethodDesc(None, [], [])

    exp_name = args.method + '_' + args.model + '_' + args.task
    saver = rst.BestSaver(lambda _: time.time(), model_name=exp_name, verbose=0)

    if args.load_model:
        if args.load_model == 'auto':
            args.load_model = saver.get_lastest_save()
        model.load_state_dict(torch.load(args.load_model, map_location='cpu'))

    if not args.no_cuda and torch.cuda.is_available():
        model = model.cuda()

    loop = rst.DefaultLoop(
        model, task,
        loss=mthd_desc.loss,
        metrics=[...] + mthd_desc.metrics,
        processors=[
            rst.Logger(exp_name), saver
        ] + mthd_desc.processors,
        batch_size=args.batch_size,
        adapter=rst.DirectPredictionAdapter(),
        optimizer='sgd',
        scheduler_base='step',
        amp=not args.no_amp
    )
    loop.scheduler = torch.optim.lr_scheduler.CyclicLR(loop.optimizer, 0.0, 0.2)
    if args.robust_test:
        loop.processors = [
            dyntrades.pgd.PGDAttack(task.metrics()[1], eps=args.eps, n_steps=20)
        ]
        loop.run(1, False)
    else:
        loop.run(args.epochs)

if __name__ == '__main__':
    main()
