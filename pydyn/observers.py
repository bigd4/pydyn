"""Observer classes for simulation monitoring.
用于模拟监控的观察者类。
"""
import cupy as cp
from ase.io import write
from datetime import datetime


class Observer:
    """Base observer for monitoring simulation progress.
    用于监控模拟进度的基础观察者。
    """
    def __init__(self, interval=1):
        self.interval = interval
        self.initialized = False

    def __call__(self, sim):
        if sim.step_count % self.interval != 0:
            return
        self.observe(sim)

    def initialize(self):
        """Initialize observer resources.
        初始化观察者资源。
        """
        self.initialized = True

    def observe(self, sim):
        """Observe simulation state at current step.
        在当前步观察模拟状态。
        """
        pass

    def finalize(self):
        """Clean up observer resources.
        清理观察者资源。
        """
        pass


class LogThermol(Observer):
    """Log thermodynamic quantities to file during simulation.
    在模拟过程中将热力学量记录到文件。
    """
    def __init__(self, filename, quantities, interval=1):
        super().__init__(interval)
        self.filename = filename
        self.quantities = quantities
        self.file = None

    def initialize(self):
        """Open log file and write header.
        打开日志文件并写入标题。
        """
        super().initialize()
        try:
            self.file = open(self.filename, "w")
            header = "Time\t" + "Step\t" + "\t".join(self.quantities.keys()) + "\n"
            self.file.write(header)
        except IOError as e:
            raise IOError(f"Failed to open log file {self.filename}: {e}")

    def observe(self, sim):
        """Write current thermodynamic values to log file.
        将当前热力学值写入日志文件。
        """
        if self.file is None:
            return
        try:
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
        except IOError as e:
            raise IOError(f"Failed to write to log file: {e}")

    def finalize(self):
        """Close log file safely.
        安全关闭日志文件。
        """
        if self.file is not None:
            try:
                self.file.close()
            except IOError:
                pass  # Best effort cleanup
            finally:
                self.file = None

    def __del__(self):
        """Ensure file is closed on object destruction.
        确保对象销毁时文件被关闭。
        """
        self.finalize()


class AtomsDump(Observer):
    """Dump atomic configurations to file at regular intervals.
    定期将原子配置导出到文件。
    """
    def __init__(self, filename, interval=10, file_format="extxyz"):
        super().__init__(interval)
        self.filename = filename
        self.file_format = file_format

    def observe(self, sim):
        """Write current atomic structure to trajectory file.
        将当前原子结构写入轨迹文件。
        """
        atoms = sim.state.to_atoms()
        write(self.filename, atoms, format=self.file_format, append=True)
