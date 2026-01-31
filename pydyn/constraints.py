import cupy as cp
from .constants import Constants


class Constraint:

    removed_dof = 0

    def apply(self, state, context):
        raise NotImplementedError


class RemoveCOMMomentum(Constraint):

    removed_dof = 3

    def apply(self, state, context):
        v_com = cp.sum(state.p, axis=0) / cp.sum(state.m)
        state.p -= state.m[:, None] * v_com
