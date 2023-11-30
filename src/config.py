from typing import Dict


class ExperimentConfig:
    def __init__(self,
                 rot_param_type: str,
                 lr: float,
                 random_rotation=None):

        if random_rotation is None:
            random_rotation = dict(z=5, y=5, x=5)

        self.rot_param_type = rot_param_type
        self.lr = lr
        self.random_rotation = random_rotation

def good_configs():
    return [
        ExperimentConfig(rot_param_type='euler', lr=1e-3), # Kinda good, didn't converge much
        ExperimentConfig(rot_param_type='quaternion', lr=1e-3), # Super good, converged to a good minima.
    ]