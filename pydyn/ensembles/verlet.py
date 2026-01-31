from .base import PositionOp, MomentumOp, Ensemble


class VelocityVerlet(Ensemble):

    def __init__(self, force_model):
        self.ops = [
            MomentumOp(force_model),
            PositionOp(),
            MomentumOp(force_model),
        ]
        self.timescale = [0.5, 1.0, 0.5]
