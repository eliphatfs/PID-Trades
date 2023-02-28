import torch_redstone as rst


class PGDAttack(rst.AdvTrainingPGD):
    def __init__(self, loss_metric, no_perturb_attrs=[], eps=0.03, step_scale=0.5, n_steps=8) -> None:
        super().__init__(loss_metric, no_perturb_attrs, eps, step_scale, n_steps, True)
