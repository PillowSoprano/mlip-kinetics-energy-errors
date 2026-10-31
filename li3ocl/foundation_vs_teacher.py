"""
The energy-error contrast of each foundation model relative to the teacher, on many more frames than DFT can afford.

    python foundation_vs_teacher.py prepare        # frames: one per single-vacancy teacher hop (weight |dq|) and 300 basin frames
    python foundation_vs_teacher.py                # base env: teacher, MACE-MP-0, CHGNet, matgl models
    conda activate mattersim && python foundation_vs_teacher.py mattersim
    python foundation_vs_teacher.py summary
    python foundation_vs_teacher.py prepare_all / base_all / mattersim_all / summary_all
                                                   # every frame of the 527 hops, weighted by |dq|, and all 1024 basin frames
    python foundation_vs_teacher.py mace_all       # one architecture, three training sets: MACE-MP-0b3 (MPtrj), MPA-0 (+sAlex), OMAT-0 (OMat24)
    python foundation_vs_teacher.py bridge [X]     # M(model - DFT) = M(model - X) on all frames + M(X - DFT) on the 211 PBE frames

M(model - DFT) = M(model - teacher) + M(teacher - DFT). The first term needs no DFT and is measured here on ~800 frames;
the second, 73.9 +- 26.7 meV from the 211 PBE single points, is common to all models. Differences between models are
therefore known much better than each model's offset from DFT.
Outputs: ../data/li3ocl_universal_frames.npz, ../data/li3ocl_universal_energies_*.csv, ../data/li3ocl_universal_contrast.csv
"""
import csv, glob, os, sys, numpy as np
from ase import Atoms

KB = 8.617333e-5; beta = 1 / (KB * 1000.0); FRAMES = "../data/li3ocl_universal_frames.npz"; FRAMES_ALL = "../data/li3ocl_universal_frames_all.npz"
M_TEACHER_DFT, SD_TEACHER_DFT = 73.9, 26.7                      # meV, foundation_vs_dft.py / mp0_vs_dft.py

def prepare():
    d = np.load("../colab/li3ocl_teacher_1000K_censored.npz"); X, Z, L = d["frames"], d["numbers"], d["cell"]
    H = np.load("../data/li3ocl_hops_T1000.npz"); li = np.where(Z == 3)[0]; rng = np.random.default_rng(7)
    def minLiLi(x):
        p = x[li]; dd = p[:, None] - p[None]; dd -= L * np.round(dd / L); r = np.linalg.norm(dd, axis=-1); r[np.diag_indices(len(li))] = 9; return r.min()
    path = []
    for h in np.unique(H["hop"]):
        k = np.where(H["hop"] == h)[0]
        if min(minLiLi(X[H["frame"][i]]) for i in k) < 1.4: continue          # fused pair somewhere in the segment: defect-state mechanism
        w = np.abs(H["dq"][k]); path.append(H["frame"][k[rng.choice(len(k), p=w / w.sum())]])
    E = np.load("../colab/step6_results/eq_1000K.npz"); eq = rng.choice(len(E["X"]), 300, replace=False)
    pos = np.concatenate([X[path], E["X"][eq]]).astype(np.float64); kind = np.array(["path"] * len(path) + ["eq"] * len(eq))
    np.savez(FRAMES, positions=pos, kind=kind, numbers=Z, cell=L)
    print(f"{len(path)} single-vacancy hop frames, {len(eq)} basin frames")

def prepare_all():
    d = np.load("../colab/li3ocl_teacher_1000K_censored.npz"); X, Z, L = d["frames"], d["numbers"], d["cell"]
    H = np.load("../data/li3ocl_hops_T1000.npz"); li = np.where(Z == 3)[0]
    def minLiLi(x):
        p = x[li]; dd = p[:, None] - p[None]; dd -= L * np.round(dd / L); r = np.linalg.norm(dd, axis=-1); r[np.diag_indices(len(li))] = 9; return r.min()
    fr, hop, w = [], [], []
    for h in np.unique(H["hop"]):
        k = np.where(H["hop"] == h)[0]
        if min(minLiLi(X[H["frame"][i]]) for i in k) < 1.4: continue
        fr += list(H["frame"][k]); hop += [h] * len(k); w += list(np.abs(H["dq"][k]) / np.abs(H["dq"][k]).sum())   # each hop carries unit weight
    E = np.load("../colab/step6_results/eq_1000K.npz")
    pos = np.concatenate([X[fr], E["X"]]).astype(np.float64); kind = np.array(["path"] * len(fr) + ["eq"] * len(E["X"]))
    np.savez(FRAMES_ALL, positions=pos, kind=kind, numbers=Z, cell=L, hop=np.array(hop + [-1] * len(E["X"])), weight=np.array(w + [1.0] * len(E["X"])))
    print(f"{len(fr)} frames of {len(set(hop))} single-vacancy hops, {len(E['X'])} basin frames")

def evaluate(calcs, out, frames=None):
    f = np.load(frames or FRAMES); Z, L, pos = f["numbers"], f["cell"], f["positions"]; cols = {}
    for name, make in calcs.items():
        calc = make(); e = []
        for x in pos:
            a = Atoms(numbers=Z, positions=x, cell=L, pbc=True); a.wrap(); a.calc = calc; e.append(a.get_potential_energy())
        cols[name] = e; print(f"{name} done", flush=True)
    with open(out, "w") as fh:
        fh.write(",".join(cols) + "\n")
        for i in range(len(pos)): fh.write(",".join(f"{cols[k][i]:.6f}" for k in cols) + "\n")

def base_models():
    from mace.calculators import mace_mp, MACECalculator
    def mgl(name):
        def f():
            import matgl
            from matgl.ext.ase import PESCalculator
            return PESCalculator(matgl.load_model(name))
        return f
    def chg():
        from chgnet.model import CHGNet
        from chgnet.model.dynamics import CHGNetCalculator
        return CHGNetCalculator(model=CHGNet.load(), use_device="cpu")
    return {"teacher": lambda: MACECalculator(model_paths="../colab/li3ocl_teacher.model", device="cpu", default_dtype="float64"),
            "MACE-MP-0-small": lambda: mace_mp(model="small", device="cpu", default_dtype="float64"),
            "MACE-MP-0-medium": lambda: mace_mp(model="medium", device="cpu", default_dtype="float64"),
            "CHGNet-MPtrj": chg, "CHGNet-MatPES-PBE": mgl("CHGNet-PES-MatPES-PBE-2025.2.10"), "M3GNet-MatPES-PBE": mgl("M3GNet-PES-MatPES-PBE-2025.2")}

def mace_data_models():
    from mace.calculators import mace_mp
    return {n: (lambda m=m: mace_mp(model=m, device="cpu", default_dtype="float64"))
            for n, m in (("MACE-MP-0b3-medium", "medium-0b3"), ("MACE-MPA-0-medium", "medium-mpa-0"), ("MACE-OMAT-0-medium", "medium-omat-0"))}

def mattersim_models():
    from mattersim.forcefield import MatterSimCalculator
    return {n: (lambda c=c: MatterSimCalculator(load_path=c, device="cpu"))
            for n, c in (("MatterSim-v1-1M", "MatterSim-v1.0.0-1M.pth"), ("MatterSim-v1-5M", "MatterSim-v1.0.0-5M.pth"))}

def summary():
    kind = np.load(FRAMES)["kind"]; P, E = kind == "path", kind == "eq"; cols = {}
    for fn in sorted(glob.glob("../data/li3ocl_universal_energies_*.csv")):
        rows = list(csv.DictReader(open(fn)))
        for k in rows[0]: cols[k] = np.array([float(r[k]) for r in rows])
    T = cols.pop("teacher"); rng = np.random.default_rng(0)
    ip = [rng.integers(0, P.sum(), P.sum()) for _ in range(2000)]; ie = [rng.integers(0, E.sum(), E.sum()) for _ in range(2000)]
    out = []
    for name, e in cols.items():
        x = 1e3 * (e - T); xp, xe = x[P], x[E]; m = xp.mean() - xe.mean()
        sd = np.std([xp[a].mean() - xe[b].mean() for a, b in zip(ip, ie)])
        mt = m + M_TEACHER_DFT; st = np.hypot(sd, SD_TEACHER_DFT)
        out.append((name, m, sd, mt, st, -1e-3 * mt * beta))
    with open("../data/li3ocl_universal_contrast.csv", "w") as fh:
        fh.write("model,M_vs_teacher_meV,sd_vs_teacher_meV,M_vs_DFT_meV,sd_vs_DFT_meV,first_order_ln_k_over_k_DFT\n")
        for r in out: fh.write(f"{r[0]}," + ",".join(f"{v:.3f}" for v in r[1:]) + "\n")
    print(f"{P.sum()} hop frames, {E.sum()} basin frames; M(teacher - DFT) = {M_TEACHER_DFT} +- {SD_TEACHER_DFT} meV")
    print(f"{'model':20s} {'M vs teacher':>16s} {'M vs DFT':>16s} {'ln(k/k_DFT)':>12s}")
    for r in sorted(out, key=lambda r: r[1]): print(f"{r[0]:20s} {r[1]:+8.1f} ± {r[2]:4.1f} {r[3]:+8.1f} ± {r[4]:4.1f} {r[5]:+12.2f}")

def summary_all():
    f = np.load(FRAMES_ALL); P = f["kind"] == "path"; E = ~P; hop, w = f["hop"][P], f["weight"][P]; cols = {}
    for fn in sorted(glob.glob("../data/li3ocl_universal_all_energies_*.csv")):
        rows = list(csv.DictReader(open(fn)))
        for k in rows[0]: cols[k] = np.array([float(r[k]) for r in rows])
    T = cols.pop("teacher"); hops = np.unique(hop); rng = np.random.default_rng(0)
    per_hop = lambda x: np.array([np.sum(w[hop == h] * x[hop == h]) for h in hops])        # |dq|-weighted mean over each hop
    bh = [rng.integers(0, len(hops), len(hops)) for _ in range(2000)]; be = [rng.integers(0, E.sum(), E.sum()) for _ in range(2000)]
    out = []
    for name, e in cols.items():
        x = 1e3 * (e - T); hp = per_hop(x[P]); xe = x[E]; m = hp.mean() - xe.mean()
        sd = np.std([hp[a].mean() - xe[b].mean() for a, b in zip(bh, be)]); mt = m + M_TEACHER_DFT
        out.append((name, m, sd, mt, np.hypot(sd, SD_TEACHER_DFT), -1e-3 * mt * beta))
    with open("../data/li3ocl_universal_contrast_all.csv", "w") as fh:
        fh.write("model,M_vs_teacher_meV,sd_vs_teacher_meV,M_vs_DFT_meV,sd_vs_DFT_meV,first_order_ln_k_over_k_DFT\n")
        for r in out: fh.write(f"{r[0]}," + ",".join(f"{v:.3f}" for v in r[1:]) + "\n")
    print(f"{P.sum()} frames of {len(hops)} hops, {E.sum()} basin frames; M(teacher - DFT) = {M_TEACHER_DFT} +- {SD_TEACHER_DFT} meV")
    print(f"{'model':20s} {'M vs teacher':>16s} {'M vs DFT':>16s} {'ln(k/k_DFT)':>12s}")
    for r in sorted(out, key=lambda r: r[1]): print(f"{r[0]:20s} {r[1]:+8.1f} ± {r[2]:4.1f} {r[3]:+8.1f} ± {r[4]:4.1f} {r[5]:+12.2f}")

def summary_bridge(bridges=("teacher", "MatterSim-v1-5M")):
    """Any potential X can carry the DFT anchor; its error bar is set by the frame-to-frame spread of X - DFT, so the
    potential closest to DFT frame by frame gives the tightest anchor at no extra DFT cost."""
    rows = list(csv.DictReader(open("../data/li3ocl_foundation_energies.csv"))); kind = np.array([r["kind"] for r in rows])
    Ed = np.array([float(r["E_dft"]) for r in rows]); D = {k: np.array([float(r[k]) for r in rows]) for k in rows[0] if k not in ("frame", "kind", "E_dft")}
    ms = list(csv.DictReader(open("../data/li3ocl_mattersim_energies.csv")))
    for k in ms[0]:
        if k != "frame": D[k] = np.array([float(r[k]) for r in ms])
    D["teacher"] = D.pop("E_teacher"); P, E = kind == "path", kind == "eq"; rng = np.random.default_rng(0)
    bp = [rng.integers(0, P.sum(), P.sum()) for _ in range(4000)]; be = [rng.integers(0, E.sum(), E.sum()) for _ in range(4000)]
    def c211(x): xp, xe = x[P], x[E]; return xp.mean() - xe.mean(), np.std([xp[a].mean() - xe[b].mean() for a, b in zip(bp, be)])
    anchor = {k: c211(1e3 * (v - Ed)) for k, v in D.items()}
    f = np.load(FRAMES_ALL); PA = f["kind"] == "path"; EA = ~PA; hop, w = f["hop"][PA], f["weight"][PA]; idx = [np.where(hop == h)[0] for h in np.unique(hop)]
    A = {}
    for fn in sorted(glob.glob("../data/li3ocl_universal_all_energies_*.csv")):
        rr = list(csv.DictReader(open(fn)))
        for k in rr[0]: A[k] = np.array([float(r[k]) for r in rr])
    bh = [rng.integers(0, len(idx), len(idx)) for _ in range(2000)]; bb = [rng.integers(0, EA.sum(), EA.sum()) for _ in range(2000)]
    def call(x):
        hp = np.array([np.sum(w[i] * x[PA][i]) for i in idx]); xe = x[EA]; return hp.mean() - xe.mean(), np.std([hp[a].mean() - xe[b].mean() for a, b in zip(bh, bb)])
    out = []
    for m in [m for m in A if m in D]:
        r = [m]
        for b in bridges:
            if m == b: r += list(anchor[b])
            else: d1, s1 = call(1e3 * (A[m] - A[b])); r += [d1 + anchor[b][0], np.hypot(s1, anchor[b][1])]
        out.append(r)
    with open("../data/li3ocl_universal_contrast_bridge.csv", "w") as fh:
        fh.write("model," + ",".join(f"M_vs_DFT_via_{b},sd_via_{b}" for b in bridges) + "\n")
        for r in out: fh.write(r[0] + "," + ",".join(f"{v:.3f}" for v in r[1:]) + "\n")
    print("anchors M(X - DFT), 211 frames:", ", ".join(f"{b} {anchor[b][0]:+.1f} ± {anchor[b][1]:.1f}" for b in bridges))
    print(f"{'model':20s}" + "".join(f"{'via ' + b:>24s}" for b in bridges) + f"{'ln(k/k_DFT) via ' + bridges[-1]:>30s}")
    for r in sorted(out, key=lambda r: r[-2]):
        print(f"{r[0]:20s}" + "".join(f"{r[1 + 2 * j]:+15.1f} ± {r[2 + 2 * j]:5.1f} " for j in range(len(bridges))) + f"{-1e-3 * beta * r[-2]:+22.2f} ± {1e-3 * beta * r[-1]:.2f}")

if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "base"
    if mode == "prepare": prepare()
    elif mode == "base": evaluate(base_models(), "../data/li3ocl_universal_energies_base.csv")
    elif mode == "mattersim": evaluate(mattersim_models(), "../data/li3ocl_universal_energies_mattersim.csv")
    elif mode == "summary": summary()
    elif mode == "prepare_all": prepare_all()
    elif mode == "base_all": evaluate(base_models(), "../data/li3ocl_universal_all_energies_base.csv", FRAMES_ALL)
    elif mode == "mattersim_all": evaluate(mattersim_models(), "../data/li3ocl_universal_all_energies_mattersim.csv", FRAMES_ALL)
    elif mode == "summary_all": summary_all()
    elif mode == "bridge": summary_bridge(("teacher", *sys.argv[2:]) if len(sys.argv) > 2 else ("teacher", "MatterSim-v1-5M"))
    elif mode == "mace_all": evaluate(mace_data_models(), "../data/li3ocl_universal_all_energies_macedata.csv", FRAMES_ALL)
