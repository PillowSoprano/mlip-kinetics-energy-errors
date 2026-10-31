"""
Batched Langevin MD of B replicas with a universal (foundation) MLIP, for Li3OCl and LGPS.

The integrator, centre-of-mass removal and hop bookkeeping are those of batched_md.py (BAOAB, BatchedMD.run, HopCounter);
only the energy-and-force call is replaced by a backend:

    MACEBackend(size)            MACE-MP-0 small or medium; neighbour list with periodic images built on the device, so the
                                 cutoff (6 A) may exceed half the cell edge (5.9 A for the 134-atom cell); one model call per step
    CHGNetBackend()              CHGNet 0.3 (MPtrj), batched through CHGNet.predict_structure
    MatGLBackend(name)           matgl models (CHGNet-, M3GNet-MatPES-PBE), one call per replica
    MatterSimBackend(ckpt)       MatterSim v1, batched through its own data loader

    eng = FoundationMD(backend, numbers, cell_lengths, positions[B, N, 3], T, device=...)
    eng.run(n_steps, callback=HopCounter(eng, ...), callback_every=5)

Every backend returns energies (eV per cell) and forces (eV/A) for all replicas; check_backend() compares them with the
package's own ASE calculator on a random frame.
"""
import itertools, numpy as np, torch
from ase import Atoms, units
from ase.data import atomic_masses
from batched_md import BatchedMD


class FoundationMD(BatchedMD):
    def __init__(self, backend, numbers, cell_lengths, positions, temperature_K, dt_fs=2.0, friction_per_fs=0.01, device="cpu",
                 dtype=torch.float32, seed=0, remove_com=True):
        self.backend = backend
        self.dev, self.dt_ = torch.device(device), dtype
        self.B, self.N = positions.shape[:2]
        self.numbers = np.asarray(numbers)
        self.L = torch.as_tensor(np.asarray(cell_lengths), dtype=dtype, device=self.dev)
        self.x = torch.as_tensor(positions, dtype=dtype, device=self.dev).clone()
        self.m = torch.as_tensor(atomic_masses[self.numbers], dtype=dtype, device=self.dev)[None, :, None]
        self.kT = units.kB * temperature_K; self.dt = dt_fs * units.fs; gam = friction_per_fs / units.fs
        self.c1 = float(np.exp(-gam * self.dt)); self.c2 = float(np.sqrt(1 - self.c1**2))
        self.gen = torch.Generator(device=self.dev); self.gen.manual_seed(seed); self.remove_com = remove_com
        self.v = torch.sqrt(self.kT / self.m) * torch.randn(self.x.shape, generator=self.gen, device=self.dev, dtype=dtype)
        self._zero_com(); self.step_count = 0; self.rebuild_every = 1
        self.backend.setup(self)
        self.e, self.f = self.energy_forces()

    def _build_neighbours(self):            # the backend owns its neighbour list
        pass

    def _needs_rebuild(self):
        return False

    def energy_forces(self):
        e, f = self.backend(self)
        return e.to(self.dev, self.dt_), f.to(self.dev, self.dt_)


def _atoms(eng, b):
    """Replica b as ASE Atoms. The integrator keeps unwrapped positions; the packages' neighbour lists expect atoms in the cell."""
    a = Atoms(numbers=eng.numbers, positions=eng.x[b].detach().cpu().double().numpy(), cell=eng.L.cpu().double().numpy(), pbc=True)
    a.wrap(); return a


class MACEBackend:
    """MACE-MP-0 with a neighbour list over the 27 nearest periodic images, rebuilt when an atom moved by half the skin."""
    def __init__(self, size="medium", skin=0.6, dtype="float32"):
        from mace.calculators import mace_mp
        self.calc = mace_mp(model=size, device="cpu", default_dtype=dtype); self.skin = skin

    def setup(self, eng):
        self.model = self.calc.models[0].to(eng.dev).to(eng.dt_).eval()
        for p in self.model.parameters(): p.requires_grad_(False)
        self.rc = float(self.model.r_max); zs = [int(z) for z in self.model.atomic_numbers]
        idx = torch.tensor([zs.index(int(z)) for z in eng.numbers])
        self.node_attrs = torch.nn.functional.one_hot(idx, len(zs)).to(eng.dt_).repeat(eng.B, 1).to(eng.dev)
        self.batch = torch.arange(eng.B, device=eng.dev).repeat_interleave(eng.N)
        self.ptr = torch.arange(0, (eng.B + 1) * eng.N, eng.N, device=eng.dev)
        self.cell = torch.diag(eng.L).repeat(eng.B, 1, 1).reshape(-1, 3)
        img = torch.tensor(list(itertools.product((-1, 0, 1), repeat=3)), dtype=eng.dt_, device=eng.dev)
        assert float(eng.L.min()) > self.rc + self.skin, "one layer of images is not enough"
        self.img = img; self._build(eng)

    def _build(self, eng):
        ei, sh = [], []
        for b in range(eng.B):                                                    # per replica keeps the memory at N^2 x 27
            d = eng.x[b][None, :, :] - eng.x[b][:, None, :]                       # d[i, j] = x_j - x_i
            s0 = -torch.round(d / eng.L)                                          # minimum image
            s = s0[:, :, None, :] + self.img[None, None]                          # [N, N, 27, 3]
            r2 = ((d[:, :, None, :] + s * eng.L) ** 2).sum(-1)
            mask = r2 < (self.rc + self.skin) ** 2
            mask &= ~((s == 0).all(-1) & torch.eye(eng.N, dtype=torch.bool, device=eng.dev)[:, :, None])
            i, j, k = mask.nonzero(as_tuple=True)
            ei.append(torch.stack([b * eng.N + i, b * eng.N + j])); sh.append(s[i, j, k])
        self.edge_index = torch.cat(ei, 1); self.unit_shifts = torch.cat(sh, 0); self.x_ref = eng.x.clone()

    def __call__(self, eng):
        if bool((((eng.x - self.x_ref) ** 2).sum(-1).max()) > (0.5 * self.skin) ** 2): self._build(eng)
        data = {"positions": eng.x.reshape(-1, 3), "node_attrs": self.node_attrs, "edge_index": self.edge_index,
                "unit_shifts": self.unit_shifts, "shifts": self.unit_shifts * eng.L, "cell": self.cell, "batch": self.batch,
                "ptr": self.ptr, "head": torch.zeros(eng.B, dtype=torch.long, device=eng.dev)}
        out = self.model(data, compute_force=True, training=False)
        return out["energy"].detach(), out["forces"].detach().reshape(eng.B, eng.N, 3)

    def ase_calc(self):
        return self.calc


class CHGNetBackend:
    """CHGNet 0.3 (MPtrj). predict_structure returns energy per atom."""
    def __init__(self, batch_size=32):
        from chgnet.model import CHGNet
        self.model = CHGNet.load(); self.bs = batch_size

    def setup(self, eng):
        from pymatgen.io.ase import AseAtomsAdaptor
        self.ad = AseAtomsAdaptor(); self.model = self.model.to(str(eng.dev)); self.dev = str(eng.dev)

    def __call__(self, eng):
        st = [self.ad.get_structure(_atoms(eng, b)) for b in range(eng.B)]
        pr = self.model.predict_structure(st, task="ef", batch_size=self.bs)
        pr = pr if isinstance(pr, list) else [pr]
        e = torch.tensor([p["e"] * eng.N for p in pr]); f = torch.tensor(np.stack([p["f"] for p in pr]))
        return e, f

    def ase_calc(self):
        from chgnet.model.dynamics import CHGNetCalculator
        return CHGNetCalculator(model=self.model, use_device="cpu")


class MatGLBackend:
    """matgl models through their ASE calculator, one replica at a time."""
    def __init__(self, name):
        import matgl
        self.name = name; self.pot = matgl.load_model(name)

    def setup(self, eng):
        from matgl.ext.ase import PESCalculator
        self.pot = self.pot.to(str(eng.dev)); self.calc = PESCalculator(self.pot)

    def __call__(self, eng):
        e, f = [], []
        for b in range(eng.B):
            a = _atoms(eng, b); a.calc = self.calc; e.append(a.get_potential_energy()); f.append(a.get_forces())
        return torch.tensor(e), torch.tensor(np.stack(f))

    def ase_calc(self):
        from matgl.ext.ase import PESCalculator
        return PESCalculator(self.pot.to("cpu"))


class MatterSimBackend:
    """MatterSim v1 (needs its own environment: torch >= 2.14)."""
    def __init__(self, ckpt="MatterSim-v1.0.0-1M.pth"):
        self.ckpt = ckpt

    def setup(self, eng):
        from mattersim.forcefield.potential import Potential
        from mattersim.datasets.utils.build import build_dataloader
        self.pot = Potential.from_checkpoint(load_path=self.ckpt, device=str(eng.dev)); self.build = build_dataloader

    def __call__(self, eng):
        atoms = [_atoms(eng, b) for b in range(eng.B)]
        dl = self.build(atoms, only_inference=True, batch_size=eng.B)
        e, f, _ = self.pot.predict_properties(dl, include_forces=True, include_stresses=False)
        return torch.tensor(np.asarray(e, dtype=float)), torch.tensor(np.stack(f))

    def ase_calc(self):
        from mattersim.forcefield import MatterSimCalculator
        return MatterSimCalculator(load_path=self.ckpt, device="cpu")


def check_backend(backend, numbers, cell_lengths, positions, device="cpu"):
    """Energies and forces of the batched backend against the package's ASE calculator for every replica."""
    eng = FoundationMD(backend, numbers, cell_lengths, positions, 1000.0, device=device)
    e, f = eng.energy_forces(); calc = backend.ase_calc(); de, df = [], []
    for b in range(eng.B):
        a = _atoms(eng, b); a.calc = calc
        de.append(abs(float(e[b]) - a.get_potential_energy())); df.append(float(np.abs(f[b].cpu().numpy() - a.get_forces()).max()))
    return max(de), max(df)
