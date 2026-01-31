import cupy as cp
from .constants import Constants


class VelocityInitializer:
    def initialize(self, state, context):
        raise NotImplementedError


class MaxwellBoltzmannDistribution(VelocityInitializer):

    def __init__(self, target_temp=None):
        self.target_temp = target_temp

    def initialize(self, state, context):
        target_temp = self.target_temp or context.target_temp
        sigma_p = cp.sqrt(
            state.m[:, None] * Constants.kB * target_temp * Constants.e_to_mv2
        )
        state.p = cp.random.standard_normal((state.N, 3)) * sigma_p
