import cupy as cp
from typing import List


class ForceModel:

    implemented_properties: List[str] = []

    def __init__(self):
        self.state = None
        self.results = {}

    def need_compute(self, state, context):
        if state.configure_same_as(self.state):
            return False
        self.state = state.copy()
        self.results = {}
        return True

