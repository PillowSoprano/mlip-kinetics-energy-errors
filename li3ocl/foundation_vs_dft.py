"""
Universal MLIPs against DFT on the Li3OCl hop: the energy-error contrast M of each foundation model on the 211 PBE
single points (131 hop-path frames, one per teacher hop drawn with weight |dq|, and 80 basin frames, 1000 K).

    python foundation_vs_dft.py [model ...]        (base env; MatterSim runs in its own env, see mattersim_vs_dft.py)

Writes ../data/li3ocl_foundation_energies.csv (one column per model, eV per cell) and ../data/li3ocl_foundation_contrast.csv
(per model: M with a bootstrap error, the frame-to-frame spread of the error on both kinds of frames, and the first-order
prediction ln(k_model/k_DFT) = -beta M), and prints the time of one energy-and-force call. The frames come from the
teacher's dynamics, which matches MACE-MP-0 (small) to 0 +- 2 meV in M; for the other models only the first-order
estimate applies, because the exponential averages of the two-sided interval need frames of one of the two potentials.
"""
import csv, os, sys, time, numpy as np
from ase.io import read

D = "../dft"; OUT = "../data/li3ocl_foundation_energies.csv"; KB = 8.617333e-5; T = 1000.0; beta = 1 / (KB * T)
d = np.load("../data/li3ocl_teacher_vs_dft.npz"); kind, frame, Ed = d["kind"], d["frame"], d["E_dft"]
P, E = kind == "path", kind == "eq"
atoms = [read(f"{D}/frames/{f}/POSCAR") for f in frame]

def mace(size):
    from mace.calculators import mace_mp
    return lambda: mace_mp(model=size, device="cpu", default_dtype="float64")
def chgnet_mptrj():
    from chgnet.model import CHGNet
    from chgnet.model.dynamics import CHGNetCalculator
    return CHGNetCalculator(model=CHGNet.load(), use_device="cpu")
def matgl_model(name):
    def f():
        import matgl
        from matgl.ext.ase import PESCalculator
        return PESCalculator(matgl.load_model(name))
    return f
MODELS = {"MACE-MP-0-small": mace("small"), "MACE-MP-0-medium": mace("medium"), "CHGNet-MPtrj": chgnet_mptrj,
          "CHGNet-MatPES-PBE": matgl_model("CHGNet-PES-MatPES-PBE-2025.2.10"), "M3GNet-MatPES-PBE": matgl_model("M3GNet-PES-MatPES-PBE-2025.2"),
          "MACE-MP-0b3-medium": mace("medium-0b3"), "MACE-MPA-0-medium": mace("medium-mpa-0"), "MACE-OMAT-0-medium": mace("medium-omat-0")}

def energies(calc):
    out = []
    t0 = time.time(); a = atoms[0].copy(); a.calc = calc; a.get_potential_energy(); a.get_forces(); t_call = time.time() - t0   # warm-up
    t0 = time.time()
    for k, a0 in enumerate(atoms):
        a = a0.copy(); a.calc = calc; out.append(a.get_potential_energy())
        if k == 9: a.get_forces(); t_call = (time.time() - t0) / 10
    return np.array(out), t_call

def summary(name, Em):
    x = beta * (Em - Ed); x -= x[E].mean()                            # constant offsets cancel
    rng = np.random.default_rng(0); xp, xe = x[P], x[E]
    bs = np.array([xp[rng.integers(0, len(xp), len(xp))].mean() - xe[rng.integers(0, len(xe), len(xe))].mean() for _ in range(2000)])
    M = xp.mean() - xe.mean()
    return dict(model=name, M_meV=1e3 * M / beta, sd_M_meV=1e3 * bs.std() / beta, sd_basin_meV=1e3 * xe.std() / beta,
                sd_path_meV=1e3 * xp.std() / beta, first_order=-M, sd_first_order=bs.std())

cols = {"frame": frame, "kind": kind, "E_dft": Ed, "E_teacher": d["E_teacher"]}
if os.path.exists(OUT):
    for r in csv.DictReader(open(OUT)):
        for k, v in r.items():
            if k not in ("frame", "kind", "E_dft", "E_teacher"): cols.setdefault(k, []).append(float(v))
MS = "../data/li3ocl_mattersim_energies.csv"
if os.path.exists(MS):
    ms = list(csv.DictReader(open(MS)))
    for k in ms[0]:
        if k != "frame" and k not in cols: cols[k] = [float(r[k]) for r in ms]
todo = sys.argv[1:] or list(MODELS)
for name in todo:
    if name in cols and name not in sys.argv[1:]: continue
    Em, tc = energies(MODELS[name]()); cols[name] = Em
    print(f"{name:20s} energy+forces of the 134-atom cell on CPU: {tc:.3f} s", flush=True)
with open(OUT, "w", newline="") as fh:
    w = csv.writer(fh); keys = list(cols); w.writerow(keys)
    for i in range(len(frame)): w.writerow([cols[k][i] if isinstance(cols[k][i], str) else f"{float(cols[k][i]):.6f}" for k in keys])
rows = [summary(n, np.array(cols[n], float)) for n in cols if n not in ("frame", "kind", "E_dft")]
with open("../data/li3ocl_foundation_contrast.csv", "w") as fh:
    fh.write(",".join(rows[0]) + "\n")
    for r in rows: fh.write(",".join(r["model"] if k == "model" else f"{r[k]:.4f}" for k in r) + "\n")
print(f"\n{'model':20s} {'M (meV)':>14s} {'sd basins':>10s} {'sd paths':>9s} {'ln(k/k_DFT) first order':>24s}")
for r in rows:
    print(f"{r['model']:20s} {r['M_meV']:+7.1f} ± {r['sd_M_meV']:4.1f} {r['sd_basin_meV']:10.0f} {r['sd_path_meV']:9.0f} {r['first_order']:+15.2f} ± {r['sd_first_order']:.2f}")
