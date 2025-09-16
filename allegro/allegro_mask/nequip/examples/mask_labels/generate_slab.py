"""Generate some toy data of a slab in an extxyz file.

Keeps the middle layer of a 3 layer slab fixed. Uses ASE's toy EMT potential.

From https://github.com/mir-group/nequip.
"""
import copy

import numpy as np

import ase
import ase.build
import ase.io
from ase.calculators.emt import EMT
from ase.visualize import view

rng = np.random.default_rng(1378)
base_atoms = ase.build.fcc100("Cu", (4, 4, 3), vacuum=10)
# Mark atoms with a different tag, i.e.
# 0 => middle layer, 1 => surface
base_atoms.set_tags(
    np.abs(base_atoms.positions[:, 2] - base_atoms.cell[2, 2] * 0.5) >= 0.1
)
view(base_atoms)
base_atoms.calc = EMT()
orig_pos = copy.deepcopy(base_atoms.positions)

for i in range(10):
    base_atoms.positions[:] = orig_pos
    base_atoms.positions += rng.normal(
        loc=0.0, scale=0.1, size=base_atoms.positions.shape
    ) * base_atoms.get_tags().reshape((-1, 1))
    #  ^ only move non-base-layer atoms
    # force a potential calculation:
    base_atoms.get_potential_energy()
    ase.io.write("Cu_EMT_slab.xyz", base_atoms, append=i > 0)
