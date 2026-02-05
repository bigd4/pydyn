import cupy as cp
import numpy as np
from ase.io import write
from datetime import datetime

class Observer:
    def __init__(self, interval=1):
        self.interval = interval
        self.initialized = False

    def __call__(self, sim):
        if sim.step_count % self.interval != 0:
            return
        self.observe(sim)

    def initialize(self):
        self.initialized = True

    def observe(self, sim):
        pass

    def finalize(self):
        pass


class LogThermol(Observer):
    def __init__(self, filename, quantities, interval=1):
        super().__init__(interval)
        self.filename = filename
        self.quantities = quantities

    def initialize(self):
        super().initialize()
        self.file = open(self.filename, "w")
        header = "Time\t" + "Step\t" + "\t".join(self.quantities.keys()) + "\n"
        self.file.write(header)

    def observe(self, sim):

        values = []
        for func in self.quantities.values():
            val = func(sim)
            if isinstance(val, cp.ndarray):
                val = float(val)
            values.append(val)
        time = datetime.now().strftime("%H:%M:%S")
        line = f"{time}\t {sim.step_count}\t" + "\t".join(f"{v:.6e}" for v in values) + "\n"
        self.file.write(line)
        self.file.flush()

    def finalize(self):
        self.file.close()


class AtomsDump(Observer):
    def __init__(self, filename, interval=10, file_format="extxyz"):
        super().__init__(interval)
        self.filename = filename
        self.file_format = file_format

    def observe(self, sim):
        atoms = sim.state.to_atoms()
        write(self.filename, atoms, format=self.file_format, append=True)
