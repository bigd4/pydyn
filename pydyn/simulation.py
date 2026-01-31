class Simulation:
    def __init__(
        self,
        state,
        context,
        dt,
        initializer,
        ensemble,
        observers=None,
    ):
        self.state = state
        self.ensemble = ensemble
        self.context = context
        self.dt = dt
        self.initializer = initializer

        self.time = 0.0
        self.step_count = 0
        self.observers = observers or []

        self._initialized = False

    def initialize(self):
        if self._initialized:
            return

        for init in self.initializer:
            init.initialize(self.state, self.context)

        for op, _ in self.ensemble.op_list:
            if hasattr(op, "extend_state"):
                op.extend_state(self.state, self.context)

        for constraint in self.context.constraints:
            constraint.apply(self.state, self.context)

        for obs in self.observers:
            obs.initialize()

        self._initialized = True

    def step(self):
        if not self._initialized:
            self.initialize()

        self.ensemble.step(self.state, self.context, self.dt)

        self.time += self.dt
        self.step_count += 1

        for obs in self.observers:
            obs(self)

    def run(self, nsteps):
        for _ in range(nsteps):
            self.step()
        for obs in self.observers:
            obs.finalize()
