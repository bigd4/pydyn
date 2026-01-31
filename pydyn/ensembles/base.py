import cupy as cp
from ..constants import Constants


class Ensemble:

    def step(self, state, context, dt):
        for op, ts in self.op_list:
            op.apply(state, context, dt * ts)
        for constraint in context.constraints:
            constraint.apply(state, context)


class Operator:

    def apply(self, state, context, dt):
        raise NotImplementedError


class PositionOp(Operator):
    def apply(self, state, context, dt):
        state.r += dt * state.p / state.m[:, None]


class MomentumOp(Operator):
    def __init__(self, force_model):
        self.force_model = force_model

    def apply(self, state, context, dt):
        self.force_model.compute(state, context)
        forces = self.force_model.results["forces"]
        state.p += dt * forces * Constants.e_to_mv2
