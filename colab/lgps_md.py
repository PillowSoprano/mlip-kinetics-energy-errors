"""
two universal potentials against each other in Li10GeP2S12, no DFT.

For every model: B replicas of the 200-atom cell (2x2x1 of mp-696128) at 800 K, Langevin BAOAB with weak friction, all
coordinates stored every SAVE steps (unwrapped, float32) with the model's own energy. Afterwards every model evaluates
the energy of every other model's stored frames, so that the first-order estimate of ln(D_A/D_B) can be computed on the
frames of either model and compared with the diffusion coefficients of both. A separate set of 50 ps runs of one model,
analysed as a single ab initio run would be (MSD from the first frame, linear fit over the last 20 ps), gives the scatter
of one such run.

    from lgps_md import *
    atoms = lgps_cell()                                     # ase.Atoms, 200 atoms, orthorhombic
    run_model(name, atoms, out_dir, B=8, ps=100)            # traj_<name>.npz
    cross_energies(name, other, out_dir)                    # cross_<name>_on_<other>.npy
    noise_runs(name, atoms, out_dir, B=16, ps=50)           # noise_<name>.npz
"""
import os, time, json, numpy as np, torch
from ase.io import read
import foundation_md as fm

T_K, DT_FS, FRICTION, SAVE = 800.0, 2.0, 0.001, 10           # friction 1/ps; frames every 20 fs
MODELS = {"MACE-MPA-0-medium": lambda B: fm.MACEBackend("medium-mpa-0"),
          "MACE-OMAT-0-medium": lambda B: fm.MACEBackend("medium-omat-0"),
          "CHGNet-MPtrj": lambda B: fm.CHGNetBackend(batch_size=B)}


def lgps_cell(cif="Li10GeP2S12_mp-696128_conventional.cif"):
    a = read(cif); a = a.repeat((2, 2, 1)); a.wrap()
    assert len(a) == 200 and np.allclose(a.cell.angles(), 90), (len(a), a.cell.angles())
    return a


def _start(atoms, B, seed):
    rng = np.random.default_rng(seed)
    return np.stack([atoms.positions + rng.normal(0, 0.03, atoms.positions.shape) for _ in range(B)])


def _engine(name, atoms, B, seed, dev):
    return fm.FoundationMD(MODELS[name](B), atoms.numbers, atoms.cell.lengths(), _start(atoms, B, seed), T_K, dt_fs=DT_FS,
                           friction_per_fs=FRICTION, device=dev, seed=seed)


def speed(name, atoms, B, dev, steps=20):
    eng = _engine(name, atoms, B, 0, dev); eng.run(3); t0 = time.time(); eng.run(steps)
    if dev == "cuda": torch.cuda.synchronize()
    return (time.time() - t0) / steps                                      # s per step for all B replicas


def run_model(name, atoms, out, B=8, ps=100, equil_ps=10, dev="cuda", seed=11, chunk=2500):
    """Equilibrate, then store every SAVE steps. Resumable: a checkpoint is written after every chunk."""
    fn, ck = f"{out}/traj_{name}.npz", f"{out}/ckpt_traj_{name}.pt"
    if os.path.exists(fn): print(name, "done"); return fn
    eng = _engine(name, atoms, B, seed, dev); X, E, Tr = [], [], []
    if os.path.exists(ck):
        st = torch.load(ck, weights_only=False); eng.x, eng.v, eng.step_count = st["x"].to(dev), st["v"].to(dev), st["step"]
        eng.e, eng.f = eng.energy_forces(); X, E, Tr = st["X"], st["E"], st["T"]; print(f"{name}: resumed at {len(X) * SAVE * DT_FS / 1e3:.1f} ps")
    else:
        eng.run(int(equil_ps * 1e3 / DT_FS)); eng.step_count = 0
    def keep(e):
        X.append(e.x.detach().cpu().numpy().astype(np.float32)); E.append(e.e.detach().cpu().numpy()); Tr.append(e.temperature())
    n_total = int(ps * 1e3 / DT_FS); t0 = time.time()
    while eng.step_count < n_total:
        eng.run(min(chunk, n_total - eng.step_count), callback=keep, callback_every=SAVE)
        torch.save({"x": eng.x.cpu(), "v": eng.v.cpu(), "step": eng.step_count, "X": X, "E": E, "T": Tr}, ck)
        print(f"{name}: {eng.step_count * DT_FS / 1e3:.1f}/{ps} ps per replica, T = {np.mean(Tr[-50:]):.0f} K, "
              f"{(time.time() - t0) / 60:.1f} min", flush=True)
    np.savez_compressed(fn, X=np.array(X), E=np.array(E), T=np.array(Tr), numbers=atoms.numbers, cell=atoms.cell.lengths(),
                        dt_fs=DT_FS, save_every=SAVE, friction_per_fs=FRICTION, T_K=T_K)
    os.remove(ck); return fn


def cross_energies(name, other, out, every=2, B=8, dev="cuda"):
    """Energy of model `name` on the stored frames of model `other` (every `every`-th frame, all replicas)."""
    fn = f"{out}/cross_{name}_on_{other}.npy"
    if os.path.exists(fn): print(fn, "done"); return fn
    d = np.load(f"{out}/traj_{other}.npz"); X = d["X"][::every]; n, R = X.shape[:2]; flat = X.reshape(n * R, *X.shape[2:])
    eng = fm.FoundationMD(MODELS[name](B), d["numbers"], d["cell"], flat[:B].astype(np.float64), T_K, device=dev)
    e = []
    for i in range(0, len(flat), B):
        x = flat[i:i + B]
        if len(x) < B: x = np.concatenate([x, flat[:B - len(x)]])
        eng.x = torch.as_tensor(x, dtype=eng.dt_, device=eng.dev)
        if hasattr(eng.backend, "_build"): eng.backend._build(eng)
        e.append(eng.energy_forces()[0].cpu().numpy())
    e = np.concatenate(e)[:len(flat)].reshape(n, R); np.save(fn, e); print(fn, e.shape); return fn


def noise_runs(name, atoms, out, B=16, ps=50, dev="cuda", seed=101):
    """B independent runs analysed like one AIMD run each: Li MSD from the first frame relative to the framework, fit over the last 20 ps."""
    fn = f"{out}/noise_{name}.npz"
    if os.path.exists(fn): print(fn, "done"); return fn
    eng = _engine(name, atoms, B, seed, dev); eng.run(int(5e3 / DT_FS)); eng.step_count = 0     # 5 ps to reach temperature (friction 1/ps)
    li, fw = np.where(atoms.numbers == 3)[0], np.where(atoms.numbers != 3)[0]
    x0 = eng.x.detach().cpu().numpy(); msd = []
    def rec(e):
        x = e.x.detach().cpu().numpy(); dx = x - x0; dx -= dx[:, fw].mean(1, keepdims=True)
        msd.append((dx[:, li] ** 2).sum(-1).mean(1))
    eng.run(int(ps * 1e3 / DT_FS), callback=rec, callback_every=SAVE)
    msd = np.array(msd); t = np.arange(1, len(msd) + 1) * SAVE * DT_FS / 1e3                     # ps
    sel = t > ps - 20; D = np.array([np.polyfit(t[sel], msd[sel, b], 1)[0] / 6 for b in range(B)]) * 1e-4   # A^2/ps -> cm^2/s
    np.savez(fn, msd=msd, t=t, D=D); print(f"{name}: D = {D.mean():.2e} cm2/s, sd of log10 D over {B} runs = {np.log10(D).std(ddof=1):.3f}")
    return fn
