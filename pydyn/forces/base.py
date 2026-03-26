import cupy as cp
from typing import List


class ForceModel:

    implemented_properties: List[str] = []

    def __init__(self):
        self.state = None
        self.results = {}

    def need_compute(self, state, context, properties):
        if state.configure_same_as(self.state):
            for prop in properties:
                if prop not in self.results:
                    return True
            return False
        self.state = state.copy()
        self.results = {}
        return True

