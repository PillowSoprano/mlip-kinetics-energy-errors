"""
Values of Figure 1 from the analysis outputs in ../data.

(a) alanine students: series estimate from their own reactive paths (ala2/model_side_all.py) against the measured total rate
    (ala2/measure_students.py); Li3OCl students: first-order estimate from their own single-vacancy hops (li3ocl/student_paths.py)
    against the single-vacancy hop rate (li3ocl/students_table.py); LGPS: first-order estimate on the frames of either potential
    and the two-sided estimate, the mean of the two ends, against the measured ratio of diffusion coefficients (lgps/lgps_estimates.py).
(b) ten universal potentials in Li3OCl: first-order estimate of ln(k/k_DFT) with the anchor carried by MatterSim-v1-5M
    (li3ocl/foundation_vs_teacher.py bridge); the part specific to each potential is the total error without that of the anchor.

    python fig_estimates_data.py          # data/fig_estimates_{ala,li,lgps1,lgps2,found}.csv
"""
import csv
import numpy as np

KB = 8.617333e-5
D = "../data"
rows = lambda f: list(csv.DictReader(open(f"{D}/{f}")))


def ala():
    est = rows("ala2_students_model_side_all.csv")
    meas = np.loadtxt(f"{D}/ala2_students_measured_all.csv", delimiter=",", skiprows=1, ndmin=2)   # rmse, total, ...
    out = []
    for r in est:
        k = np.argmin(abs(meas[:, 0] - float(r["force_rmse_meVA"])))                              # students are keyed by force error
        out.append((r["student"], f"{float(r['series']):.2f}", f"{float(r['sd_series']):.2f}".rstrip("0").rstrip("."), f"{meas[k, 1]:.2f}", "0"))
    return out


def li():
    meas = {r["model"]: r for r in rows("li3ocl_students_1000K.csv")}
    return [(r["student"], f"{float(r['first_order']):.2f}", f"{float(r['sd_first_order']):.2f}",
             f"{float(meas[r['student']]['clean_measured']):.2f}", f"{float(meas[r['student']]['clean_se']):.2f}")
            for r in rows("li3ocl_student_paths_1000K.csv") if r["ensemble"] == "single"]


SHORT = {"MACE-MPA-0-medium": "MPA", "MACE-OMAT-0-medium": "OMAT", "CHGNet-MPtrj": "CHG"}


def lgps():
    p = {(r["frames_of"], r["other"]): r for r in rows("lgps_prediction.csv")}
    one = [(f"{SHORT[o]}/{SHORT[s]}", f"{float(r['first_order']):.3f}", f"{float(r['se_first']):.3f}",
            f"{float(r['measured']):.3f}", f"{float(r['se_measured']):.3f}") for (s, o), r in p.items()]
    two = []
    for s, o in (("MACE-MPA-0-medium", "CHGNet-MPtrj"), ("MACE-OMAT-0-medium", "CHGNet-MPtrj"), ("MACE-MPA-0-medium", "MACE-OMAT-0-medium")):
        f, b = p[s, o], p[o, s]                                                    # ln(D_o/D_s) from the frames of s, and of o (sign reversed)
        m = 0.5 * (float(f["first_order"]) - float(b["first_order"])); e = 0.5 * np.hypot(float(f["se_first"]), float(b["se_first"]))
        two.append((f"{SHORT[o]}/{SHORT[s]}", f"{m:.3f}", f"{e:.3f}", f"{float(f['measured']):.3f}", f"{float(f['se_measured']):.3f}"))
    return one, two


NAMES = {"MACE-MP-0-small": "MACE-MP-0 small", "MACE-MP-0b3-medium": "MACE-MP-0b3", "MACE-MP-0-medium": "MACE-MP-0 medium",
         "M3GNet-MatPES-PBE": "M3GNet-MatPES", "CHGNet-MatPES-PBE": "CHGNet-MatPES", "CHGNet-MPtrj": "CHGNet",
         "MACE-OMAT-0-medium": "MACE-OMAT-0", "MatterSim-v1-5M": "MatterSim 5M", "MatterSim-v1-1M": "MatterSim 1M",
         "MACE-MPA-0-medium": "MACE-MPA-0"}


def found():
    beta = 1e-3 / (KB * 1000.0)                                                    # per meV at 1000 K
    b = {r["model"]: r for r in rows("li3ocl_universal_contrast_bridge.csv")}
    anchor = float(b["MatterSim-v1-5M"]["sd_via_MatterSim-v1-5M"])
    sel = sorted((m for m in NAMES), key=lambda m: -float(b[m]["M_vs_DFT_via_MatterSim-v1-5M"]))
    out = []
    for y, m in enumerate(sel):
        M, sd = float(b[m]["M_vs_DFT_via_MatterSim-v1-5M"]), float(b[m]["sd_via_MatterSim-v1-5M"])
        own = np.sqrt(max(sd ** 2 - anchor ** 2, 0.0))
        out.append((str(y), NAMES[m], f"{-beta * M:.3f}", f"{beta * sd:.3f}", f"{beta * own:.3f}", f"{M:.1f}", f"{sd:.1f}"))
    return out


def write(name, header, data):
    with open(f"data/{name}", "w") as fh:
        fh.write(header + "\n" + "".join(",".join(r) + "\n" for r in data))


if __name__ == "__main__":
    H = "name,pred,epred,meas,emeas"
    write("fig_estimates_ala.csv", H, ala())
    write("fig_estimates_li.csv", H, li())
    one, two = lgps()
    write("fig_estimates_lgps1.csv", H, one)
    write("fig_estimates_lgps2.csv", H, two)
    write("fig_estimates_found.csv", "y,name,lnk,etot,eown,M,sdM", found())
