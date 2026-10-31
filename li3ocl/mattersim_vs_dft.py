"""
MatterSim on the 211 Li3OCl PBE single points (companion of foundation_vs_dft.py, which needs a different torch).

    conda activate mattersim && python mattersim_vs_dft.py

Writes ../data/li3ocl_mattersim_energies.csv with one column per MatterSim checkpoint (eV per cell); the summary is
printed by foundation_vs_dft.py, which reads this file when it exists.
"""
import time, numpy as np
from ase.io import read
from mattersim.forcefield import MatterSimCalculator

D = "../dft"
d = np.load("../data/li3ocl_teacher_vs_dft.npz"); frame = d["frame"]
atoms = [read(f"{D}/frames/{f}/POSCAR") for f in frame]
cols = {}
for name, ckpt in (("MatterSim-v1-1M", "MatterSim-v1.0.0-1M.pth"), ("MatterSim-v1-5M", "MatterSim-v1.0.0-5M.pth")):
    calc = MatterSimCalculator(load_path=ckpt, device="cpu")
    a = atoms[0].copy(); a.calc = calc; a.get_potential_energy(); a.get_forces()
    t0 = time.time(); e = []
    for k, a0 in enumerate(atoms):
        a = a0.copy(); a.calc = calc; e.append(a.get_potential_energy())
        if k == 9: a.get_forces(); tc = (time.time() - t0) / 10
    cols[name] = np.array(e); print(f"{name:20s} energy+forces of the 134-atom cell on CPU: {tc:.3f} s", flush=True)
with open("../data/li3ocl_mattersim_energies.csv", "w") as fh:
    fh.write("frame," + ",".join(cols) + "\n")
    for i, f in enumerate(frame): fh.write(f + "," + ",".join(f"{cols[k][i]:.6f}" for k in cols) + "\n")
