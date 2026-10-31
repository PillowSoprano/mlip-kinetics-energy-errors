"""
Li diffusion coefficient of each model in LGPS at 800 K from the stored trajectories, and the scatter of a
single 50 ps run (noise runs of MACE-MPA-0).

D per replica from the Li mean-square displacement relative to the framework centre of mass, averaged over time origins,
fitted on lags LAG0-LAG1 ps. Error of ln D from the scatter over the 8 replicas.

    python lgps_diffusion.py            # prints the table, writes ../data/lgps_D.csv
"""
import numpy as np

RES = "../colab/lgps_results"
MODELS = ["MACE-MPA-0-medium", "MACE-OMAT-0-medium", "CHGNet-MPtrj"]
LAG0, LAG1 = 5.0, 20.0                                                                  # ps

def li_msd(X, Z, dt_ps, lags):
    li, fw = Z == 3, Z != 3
    y = X[:, :, li] - X[:, :, fw].mean(2, keepdims=True)                               # (frames, replicas, Li, 3), framework drift removed
    out = np.zeros((len(lags), X.shape[1]))
    for k, n in enumerate(lags):
        d = y[n:] - y[:-n]; out[k] = (d ** 2).sum(-1).mean((0, 2))
    return out

def model_D(name):
    d = np.load(f"{RES}/traj_{name}.npz"); X, Z = d["X"].astype(np.float64), d["numbers"]; dt = float(d["dt_fs"]) * int(d["save_every"]) / 1e3
    jump = np.abs(np.diff(X, axis=0)).max(); assert jump < 3, f"{name}: coordinates are wrapped (largest step {jump:.1f} A)"
    lags = np.unique(np.round(np.linspace(LAG0, LAG1, 16) / dt).astype(int)); msd = li_msd(X, Z, dt, lags)
    D = np.array([np.polyfit(lags * dt, msd[:, b], 1)[0] / 6 for b in range(X.shape[1])]) * 1e-4    # A^2/ps -> cm^2/s
    return D, X.shape[0] * dt, d["T"].mean()

if __name__ == "__main__":
    rows = []
    print(f"{'model':20s} {'ps x replicas':>14s} {'<T> (K)':>8s} {'D (cm2/s)':>10s} {'ln D':>8s} {'se ln D':>8s}")
    for m in MODELS:
        D, ps, T = model_D(m); lnD = np.log(D.mean()); se = np.std(np.log(D), ddof=1) / np.sqrt(len(D))
        rows.append((m, ps, len(D), T, D.mean(), lnD, se)); print(f"{m:20s} {ps:7.0f} x {len(D):<5d} {T:8.1f} {D.mean():10.2e} {lnD:8.3f} {se:8.3f}")
    n = np.load(f"{RES}/noise_MACE-MPA-0-medium.npz"); Dn = n["D"]
    print(f"\nnoise runs (MACE-MPA-0, {len(Dn)} x 50 ps, MSD from the first frame, fit over the last 20 ps):")
    print(f"  D = {Dn.mean():.2e} cm2/s;  sd of ln D of one run = {np.std(np.log(Dn), ddof=1):.3f}  (log10: {np.std(np.log10(Dn), ddof=1):.3f});  range {Dn.min():.2e}-{Dn.max():.2e}")
    print("\nmeasured ln(D_A/D_B):")
    for i in range(3):
        for j in range(i + 1, 3):
            a, b = rows[i], rows[j]; print(f"  {a[0]:20s} / {b[0]:20s} {a[5] - b[5]:+.3f} ± {np.hypot(a[6], b[6]):.3f}")
    with open("../data/lgps_D.csv", "w") as fh:
        fh.write("model,ps_per_replica,replicas,T_mean_K,D_cm2_s,lnD,se_lnD\n")
        for r in rows: fh.write(",".join(str(v) for v in r) + "\n")
