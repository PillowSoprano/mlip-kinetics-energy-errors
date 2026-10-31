"""
prediction of ln(D_A/D_B) from the frames of one model, compared with the measured ratio.

Sites are not assumed. For each model, the Li density in the framework frame (all stored frames, Gaussian smoothing) is cut into
basins by steepest ascent; basins separated by less than B_MERGE kT are merged (persistence order). The core of a basin is the
part within C_CORE kT of its minimum of F = -ln(density). An ion is committed to the last core it visited; a hop is the entry
into a different core, and its reactive segment runs from the last frame in the old core to the first frame in the new one.
Along each segment the progress coordinate s is the projection on the vector between the two core centres. F(s) is taken from
the model's equilibrium density along that line (largest density within TUBE of it), giving the 1D committor q(s) ~ int exp(F)
and the committor increment |dq| per frame, the path representation of nu used in li3ocl/analyze_hops.py.
The weight of a frame is the sum of |dq| of all hops in progress, so with many ions hopping at once the estimate is still the
per-hop change of the barrier.

    ln(D_other/D_self) ~ -beta (<dV>_nu - <dV>_mu) + beta^2/2 (var_nu dV - var_mu dV),   dV = V_other - V_self on the frames of `self`

<.>_mu is over all frames; the first term alone is the first-order estimate. Errors: bootstrap over blocks of BLOCK_PS of each replica.

    python lgps_estimates.py                 # all ordered pairs, ../data/lgps_prediction.csv
    python lgps_estimates.py hops            # hop statistics only, ../data/lgps_hops.csv
"""
import sys, numpy as np
from scipy import ndimage

RES = "../colab/lgps_results"
MODELS = ["MACE-MPA-0-medium", "MACE-OMAT-0-medium", "CHGNet-MPtrj"]
BIN, SMOOTH = 0.2, 0.3                      # A
B_MERGE, C_CORE = 1.0, 1.0                  # kT
NBIN_S, TUBE, BLOCK_PS, NBOOT = 40, 1.0, 5.0, 2000
KB = 8.617333e-5


def load(name):
    d = np.load(f"{RES}/traj_{name}.npz"); X, Z, L = d["X"].astype(np.float64), d["numbers"], d["cell"]
    li, fw = Z == 3, Z != 3
    y = X[:, :, li] - (X[:, :, fw].mean(2, keepdims=True) - X[:1, :, fw].mean(2, keepdims=True))     # Li in the framework frame, unwrapped
    dt = float(d["dt_fs"]) * int(d["save_every"]) / 1e3
    return y, L, dt, d["E"].astype(np.float64), float(d["T_K"])


def basins(y, L):
    """Basin label of every grid voxel, core mask, core centres, and the grid geometry."""
    nb = np.round(L / BIN).astype(int); h = L / nb
    H, _ = np.histogramdd((y % L).reshape(-1, 3), bins=nb, range=[(0, l) for l in L])
    rho = ndimage.gaussian_filter(H, SMOOTH / h, mode="wrap") + 1e-12; F = -np.log(rho / rho.max())
    # steepest ascent: every voxel points to its densest neighbour (27-neighbourhood, periodic), then pointer jumping
    flat = np.arange(rho.size).reshape(nb); best, bval = flat.copy(), rho.copy()
    for s in np.ndindex(3, 3, 3):
        sh = tuple(np.array(s) - 1); r = np.roll(rho, sh, (0, 1, 2)); i = np.roll(flat, sh, (0, 1, 2)); m = r > bval; bval[m] = r[m]; best[m] = i[m]
    ptr = best.ravel()
    while True:
        nxt = ptr[ptr]
        if (nxt == ptr).all(): break
        ptr = nxt
    peaks, lab = np.unique(ptr, return_inverse=True); Ff = F.ravel(); fmin = Ff[peaks].copy()
    # persistence merging over 6-neighbour boundaries: saddle of a boundary pair = max(F_u, F_v)
    pairs = []
    for ax in range(3):
        nbr = np.roll(flat, -1, ax).ravel(); a, b = lab, lab[nbr]; m = a != b
        pairs.append(np.c_[a[m], b[m], np.maximum(Ff[m], Ff[nbr][m])])
    P = np.concatenate(pairs); P = P[np.argsort(P[:, 2])]
    par = np.arange(len(peaks)); depth_min = fmin.copy()
    def root(i):
        while par[i] != i: par[i] = par[par[i]]; i = par[i]
        return i
    for a, b, s in P:
        ra, rb = root(int(a)), root(int(b))
        if ra == rb: continue
        shallow = ra if depth_min[ra] > depth_min[rb] else rb; deep = rb if shallow == ra else ra
        if s - depth_min[shallow] < B_MERGE: par[shallow] = deep                      # the shallower basin is absorbed
    R = np.array([root(i) for i in range(len(peaks))]); site, lab = np.unique(R[lab], return_inverse=True)
    fmin_site = np.array([Ff[lab == k].min() for k in range(len(site))])
    core = Ff < fmin_site[lab] + C_CORE
    # core centres (density-weighted, minimum image about the core's densest voxel)
    g = np.stack(np.meshgrid(*[(np.arange(n) + .5) * hh for n, hh in zip(nb, h)], indexing="ij"), -1).reshape(-1, 3); w = rho.ravel()
    cen = np.zeros((len(site), 3))
    for k in range(len(site)):
        m = core & (lab == k); ref = g[m][np.argmax(w[m])]; u = g[m] - ref; u -= L * np.round(u / L); cen[k] = (ref + (u * w[m, None]).sum(0) / w[m].sum()) % L
    return lab, core, cen, nb, h, F


def hops(y, L, dt):
    lab, core, cen, nb, h, F = basins(y, L)
    T, R, N, _ = y.shape
    v = np.floor((y % L) / h).astype(int) % nb; vi = np.ravel_multi_index((v[..., 0], v[..., 1], v[..., 2]), nb)
    inc = np.where(core[vi], lab[vi], -1)                                                     # core label per frame, -1 outside cores
    seg = []                                                                                   # (replica, ion, f0, f1, A, B)
    for r in range(R):
        for i in range(N):
            c = inc[:, r, i]; idx = np.where(c >= 0)[0]
            if len(idx) == 0: continue
            cc = c[idx]; ch = np.where(cc[1:] != cc[:-1])[0]
            for k in ch: seg.append((r, i, idx[k], idx[k + 1], cc[k], cc[k + 1]))          # last frame in core A, first in core B
    seg = np.array(seg)
    # progress coordinate s along the core-centre vector; committor q(s) ~ int exp(F) along the tube between the two cores,
    # F(s) = -ln of the largest smoothed density within TUBE A of the line (equilibrium density of this model)
    sk = np.linspace(0, 1, NBIN_S + 1); prof = {}
    def q_of(A, B):
        if (A, B) not in prof:
            e = cen[B] - cen[A]; e -= L * np.round(e / L); n1 = np.cross(e, [0, 0, 1.0] if abs(e[2]) < .9 * np.linalg.norm(e) else [1.0, 0, 0])
            n1 /= np.linalg.norm(n1); n2 = np.cross(e, n1); n2 /= np.linalg.norm(n2)
            off = np.array([r * (np.cos(a) * n1 + np.sin(a) * n2) for r in np.arange(0, TUBE + 1e-9, 0.2) for a in np.linspace(0, 2 * np.pi, 12, endpoint=False)])
            p = (cen[A] + sk[:, None, None] * e + off[None]) % L; v = np.floor(p / h).astype(int) % nb
            Fs = F[v[..., 0], v[..., 1], v[..., 2]].min(1); x = np.exp(Fs - Fs.max()); q = np.r_[0, np.cumsum(0.5 * (x[1:] + x[:-1]))]
            prof[A, B] = (q / q[-1], Fs.max() - max(Fs[0], Fs[-1]))
        return prof[A, B]
    W = np.zeros((T, R)); S, bar = [], []
    for r, i, f0, f1, A, B in seg:
        e = cen[B] - cen[A]; e -= L * np.round(e / L); u = y[f0:f1 + 1, r, i] - cen[A]; u -= L * np.round(u / L); s = u @ e / (e @ e); S.append(s)
        qk, b = q_of(A, B); q = np.interp(np.clip(s, 0, 1), sk, qk); g = 0.5 * (np.r_[q[1:], q[-1]] - np.r_[q[0], q[:-1]]); W[f0:f1 + 1, r] += np.abs(g); bar.append(b)
    Fs = np.array(bar)
    jump = np.array([np.linalg.norm((lambda e: e - L * np.round(e / L))(cen[B] - cen[A])) for *_, A, B in seg])
    transit = (seg[:, 3] - seg[:, 2] + 1).sum() / (T * R)                                       # ions between cores per frame
    return dict(seg=seg, W=W, Fs=Fs, n_sites=len(cen), jump=jump, dur=(seg[:, 3] - seg[:, 2]) * dt, T=T, R=R, N=N, dt=dt,
                peak_mean=1 / np.exp(-F).mean(), transit=transit)


def contrast(self, other, H, E_self, beta):
    """First- and second-order estimate of ln(D_other/D_self) on the frames of `self` (every 2nd stored frame carries the cross
    energy). dV is centred on each replica's mean, which leaves <dV>_nu - <dV>_mu unchanged in expectation and drops slow offsets
    between replicas. Block bootstrap over (replica, BLOCK_PS) blocks."""
    X = np.load(f"{RES}/cross_{other}_on_{self}.npy").astype(np.float64); n = X.shape[0]
    dV = X - E_self[::2][:n]; dV -= dV.mean(0)                                                   # (frames, replicas), eV
    W = H["W"][0::2][:n] + np.r_[H["W"][1::2], np.zeros((1, H["R"]))][:n]                         # odd frames lumped onto the preceding even frame
    blk = int(round(BLOCK_PS / (2 * H["dt"]))); nblk = n // blk; dV, W = dV[:nblk * blk], W[:nblk * blk]
    def est(idx):                                                                                # idx: list of (block, replica)
        d = np.concatenate([dV[b * blk:(b + 1) * blk, r] for b, r in idx]); w = np.concatenate([W[b * blk:(b + 1) * blk, r] for b, r in idx])
        m_nu = (w * d).sum() / w.sum(); v_nu = (w * (d - m_nu) ** 2).sum() / w.sum()
        first = -beta * (m_nu - d.mean()); return first, first + 0.5 * beta ** 2 * (v_nu - d.var())
    units = [(b, r) for b in range(nblk) for r in range(H["R"])]; rng = np.random.default_rng(0)
    p = est(units); bs = np.array([est([units[k] for k in rng.integers(0, len(units), len(units))]) for _ in range(NBOOT)])
    return p[0], bs[:, 0].std(), p[1], bs[:, 1].std(), beta * dV.std()


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "predict"
    meas = {r.split(",")[0]: (float(r.split(",")[5]), float(r.split(",")[6])) for r in open("../data/lgps_D.csv").read().split("\n")[1:] if r}
    HOP, EN = {}, {}
    for m in MODELS:
        y, L, dt, E, TK = load(m); H = hops(y, L, dt); HOP[m], EN[m] = H, E
        nh = len(H["seg"]); ps = H["T"] * dt * H["R"]
        print(f"{m:20s} {H['n_sites']:3d} sites; {nh} hops = {nh / ps / H['N']:.2f} per Li per ps; jump length median {np.median(H['jump']):.2f} A "
              f"(10-90%: {np.quantile(H['jump'], .1):.2f}-{np.quantile(H['jump'], .9):.2f}); segment median {np.median(H['dur']) * 1e3:.0f} fs, "
              f"90% below {np.quantile(H['dur'], .9) * 1e3:.0f} fs; tube barrier median {np.median(H['Fs']):.2f} kT", flush=True)
    if mode == "hops":
        # Table of the Supporting Information: correlation factor 6D / (Gamma <l^2>), Gamma the hop rate per ion
        D = {r.split(",")[0]: float(r.split(",")[4]) for r in open("../data/lgps_D.csv").read().split("\n")[1:] if r}
        with open("../data/lgps_hops.csv", "w") as fh:
            fh.write("model,basins,peak_over_mean,hops_per_Li_ps,median_jump_A,mean_sq_jump_A2,median_segment_fs,ions_in_transit,correlation_factor\n")
            for m, H in HOP.items():
                G = len(H["seg"]) / (H["T"] * H["dt"] * H["R"]) / H["N"]; l2 = (H["jump"] ** 2).mean(); f = 6 * D[m] * 1e4 / (G * l2)   # cm2/s -> A2/ps
                fh.write(f"{m},{H['n_sites']},{H['peak_mean']:.1f},{G:.3f},{np.median(H['jump']):.2f},{l2:.2f},{np.median(H['dur']) * 1e3:.0f},{H['transit']:.1f},{f:.3f}\n")
                print(f"{m:20s} peak/mean {H['peak_mean']:.0f}; <l^2> {l2:.2f} A^2; ions in transit {H['transit']:.1f}; correlation factor {f:.2f}")
        sys.exit()
    beta = 1 / (KB * 800.0); rows = []
    print(f"\n{'frames of':20s} {'other':20s} {'beta sd dV':>10s} {'first order':>14s} {'+ second order':>15s} {'measured':>14s}")
    for s in MODELS:
        for o in MODELS:
            if s == o: continue
            p1, e1, p2, e2, bs = contrast(s, o, HOP[s], EN[s], beta); ms = meas[o][0] - meas[s][0], np.hypot(meas[o][1], meas[s][1])
            rows.append((s, o, p1, e1, p2, e2, *ms, bs)); print(f"{s:20s} {o:20s} {bs:10.1f} {p1:+7.2f} ± {e1:.2f} {p2:+8.2f} ± {e2:.2f} {ms[0]:+7.2f} ± {ms[1]:.2f}")
    with open("../data/lgps_prediction.csv", "w") as fh:
        fh.write("frames_of,other,first_order,se_first,second_order,se_second,measured,se_measured,beta_sd_dV\n")
        for r in rows: fh.write(",".join(str(x) for x in r) + "\n")
