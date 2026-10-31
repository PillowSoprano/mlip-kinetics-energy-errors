"""
Second-order term of the hop-rate estimate for the universal potentials on the teacher's frames.

On the frames of the teacher (3540 frames of 528 single-vacancy hops, weighted by |dq| within each hop, and 1024 basin frames),
with dV = E_P - E_teacher,

    ln(k_P/k_teacher) ~ -beta (<dV>_nu - <dV>_basins) + beta^2/2 (Var_nu dV - Var_basins dV),

the second term being the curvature with the committor held fixed. Its size against the first term says whether the first-order
estimate on third-party frames is enough. Errors: bootstrap over hops and basin frames.

    python foundation_second_order.py    # ../data/li3ocl_universal_second_order.csv
"""
import csv, glob, numpy as np

KB = 8.617333e-5; beta = 1 / (KB * 1000.0); NBOOT = 2000

if __name__ == "__main__":
    f = np.load("../data/li3ocl_universal_frames_all.npz"); P = f["kind"] == "path"; hop, w = f["hop"][P], f["weight"][P]
    hops = np.unique(hop); idx = [np.where(hop == h)[0] for h in hops]; wn = [w[i] / w[i].sum() for i in idx]
    A = {}
    for fn in sorted(glob.glob("../data/li3ocl_universal_all_energies_*.csv")):
        rows = list(csv.DictReader(open(fn)))
        for k in rows[0]: A[k] = np.array([float(r[k]) for r in rows])
    T = A.pop("teacher"); rng = np.random.default_rng(0); nb = (~P).sum()
    bh = [rng.integers(0, len(hops), len(hops)) for _ in range(NBOOT)]; bb = [rng.integers(0, nb, nb) for _ in range(NBOOT)]

    def est(x, hsel, bsel):
        xp = np.concatenate([x[P][idx[h]] for h in hsel]); wp = np.concatenate([wn[h] for h in hsel]); wp = wp / wp.sum()
        xb = x[~P][bsel]; m_nu = (wp * xp).sum(); v_nu = (wp * (xp - m_nu) ** 2).sum()
        first = -(m_nu - xb.mean()); return first, 0.5 * (v_nu - xb.var())

    out = []
    for name, e in A.items():
        x = beta * (e - T)                                                      # beta dV, dimensionless
        p = est(x, range(len(hops)), np.arange(nb)); bs = np.array([est(x, a, b) for a, b in zip(bh, bb)])
        out.append((name, p[0], bs[:, 0].std(), p[1], bs[:, 1].std(), x[P].std(), x[~P].std()))
    out.sort(key=lambda r: r[1])
    print(f"{'potential':22s} {'first order':>14s} {'second-order term':>18s} {'beta sd dV paths / basins':>26s}")
    for r in out: print(f"{r[0]:22s} {r[1]:+7.2f} ± {r[2]:.2f} {r[3]:+10.2f} ± {r[4]:.2f} {r[5]:13.2f} / {r[6]:.2f}")
    with open("../data/li3ocl_universal_second_order.csv", "w") as fh:
        fh.write("model,first_order,sd_first,second_order_term,sd_second,beta_sd_dV_paths,beta_sd_dV_basins\n")
        for r in out: fh.write(f"{r[0]}," + ",".join(f"{v:.4f}" for v in r[1:]) + "\n")
