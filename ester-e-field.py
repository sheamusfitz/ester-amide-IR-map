"""this is not the one I actually used. see efield-O-C=O.py for the correct file."""

import MDAnalysis as mda
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import norm
import lmfit as lf
import time
import os


plt.style.use(
    "https://raw.githubusercontent.com/sheamusfitz/mpl-styles/main/smallfigs.mplstyle")

import sys
sys.path.insert(0, '/Users/shea/dcuments/custom-python-scripts/')
from weighted_unc import weighted_unc
from acorcor.acorcor import corrected_exp, corrected_strex
from acorcor import autocor



system = 'PSM-75'


'''
- okay so i want the electric field function to take a residue number and
the atomtypes of the two things i want to use, as well as the list of other 
atoms to ignore.
'''

def compute_electric_field(ts, universe, resid: int, atom0: str, atom1: str, cutoff, 
                           atoms_to_ignore: str='none'):
    """
    atom0: the one we want the electric field calculated at. this is actually the atomselect string
        for MDAnalysis
    atom1: the other one in the pair with atom0
    atoms_to_ignore: another atomselect string, for any other atoms
    """
    
    e_field_at_a0 = np.zeros(3)
    universe.trajectory[ts]
    a0 = universe.select_atoms(f'resid {resid} and name {atom0}')
    a1 = universe.select_atoms(f'resid {resid} and name {atom1}')
    
    bond_vector = a1.positions[0] - a0.positions[0]
    unit_vector = bond_vector / np.linalg.norm(bond_vector)
    
    if atoms_to_ignore == 'none':
        solvent_atoms = universe.select_atoms(f'around {cutoff} id {" ".join(map(str, a0.ids.flat))}')
    else:
        solvent_atoms = universe.select_atoms(
            f'(around {cutoff} id {" ".join(map(str, a0.ids.flat))}) and not {atoms_to_ignore}'
        )
    print(len(solvent_atoms))
    # print(solvent_atoms)
    for atom in solvent_atoms:
        r_i = atom.position - a0.positions[0]
        # rw = r_i.copy()
        r_i -= np.round(r_i / universe.dimensions[:3]) * universe.dimensions[:3]  # PBC fix
        # if np.sum(rw) != np.sum(r_i):
        #     print(rw, r_i)
        distance_i = np.linalg.norm(r_i)
        r_i_unit = r_i / distance_i
        # distance_jn_bohr = distance_jn / bohr_to_angstrom
        q_i = atom.charge
        e_field_contribution = (q_i * r_i_unit) / (distance_i ** 2)
        e_field_at_a0 += e_field_contribution

    e_projection = np.dot(e_field_at_a0, unit_vector)
    return e_projection


universe = mda.Universe(f'../../psm-2dir/8nm/{system}/gromacs/step5_input.psf', 
                        f'../../psm-2dir/8nm/{system}/gromacs/production.xtc')
alldppc = universe.select_atoms('resname DPPC')

should_run = True
# if os.path.isfile(f'../data/{system}-efield.npy'):
#     if os.path.getsize(f'../data/{system}-efield.npy') > 0:
#         should_run = False

if should_run:
    efield = np.zeros((len(alldppc.residues.resids), len(universe.trajectory)))
    
    # short_n = 2
    # efield = np.zeros((short_n, len(universe.trajectory)))
    
    tstart = time.time()
    t2last = tstart
    print(tstart)
    for ts in universe.trajectory:
        t = ts.frame
        if not t%10:
            print('.',end='', flush=True)
        if not t%100:
            tlast = time.time()
            print(f"{t/len(universe.trajectory)*100: 5.1f}%", end=' ')
            print(f"\tlast step: {tlast - t2last:0.3f}s\tremain: {(tlast-tstart)*(13337-t)/max(t, 1)/60:0.2f}mins")
            print(efield[0, max(t-1,0)])
            t2last = tlast
        for i, resid in enumerate(alldppc.residues.resids):
            if i >= len(efield):
                if not t%100:
                    print('|', end='')
                break
            efield[i, t] = compute_electric_field(t, universe, resid, 'C21', 'O22', 
                                                      30, f'resid {resid} and name O21')
    with open(f'../data/{system}-efield.npy', 'wb') as f:
        np.save(f, efield)