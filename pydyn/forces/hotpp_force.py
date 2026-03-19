import numpy as np
import torch
from .base import ForceModel
import cupy as cp
import torch.utils.dlpack as torch_dlpack


# 你是真牛逼大发了，from是from，to也是from是吧，你妈的
def cp_to_torch(x):
    return torch_dlpack.from_dlpack(cp.from_dlpack(x))


def torch_to_cp(x):
    return cp.from_dlpack(torch_dlpack.to_dlpack(x))


class MiaoForceModel(ForceModel):

    def __init__(
        self,
        neighbor_list,
        model_file: str = "model.pt",
        spin: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()
        self.model = torch.jit.load(model_file).cuda()
        self.cutoff = float(self.model.cutoff.detach().cpu().numpy())
        self.spin = spin
        self.neighbor_list = neighbor_list

    def compute(self, state, context, properties=["energy", "forces", "virial"]):
        if not self.need_compute(state, context, properties):
            return
        idx_i, idx_j, offset = self.neighbor_list.find_neighbor(state)
        data = {
            "atomic_number": cp_to_torch(state.atomic_number),
            "idx_i": cp_to_torch(idx_i),
            "idx_j": cp_to_torch(idx_j),
            "coordinate": cp_to_torch(state.r).to(torch.float32),
            "n_atoms": torch.tensor([state.N], dtype=torch.long, device="cuda"),
            "offset": cp_to_torch(offset).to(torch.float32),
            "scaling": torch.eye(3, dtype=torch.float32, device="cuda").view(1, 3, 3),
            "batch": torch.zeros(state.N, dtype=torch.long, device="cuda"),
            "volume": torch.tensor(
                [state.volume.item()], dtype=torch.float32, device="cuda"
            ),
        }
        if self.spin:
            data["spin"] = cp_to_torch(state.spin.spins).to(torch.float32)

        data = self.model(data, properties, create_graph=False)
        if "energy" in properties:
            self.results["potential_energy"] = torch_to_cp(data["energy_p"][0]).astype(
                cp.float64
            )
        if "forces" in properties:
            self.results["forces"] = torch_to_cp(data["forces_p"]).astype(cp.float64)
        if "virial" in properties:
            # stress = torch_to_cp(data["stress_p"][0]).astype(cp.float64)
            # self.results["virial"] = -state.volume * stress
            self.results["virial"] = torch_to_cp(data["virial_p"][0]).astype(cp.float64)

        if "spin_torques" in properties:
            self.results["spin_torques"] = torch_to_cp(data["spin_torques_p"]).astype(
                cp.float64
            )
