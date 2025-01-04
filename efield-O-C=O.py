# import scipy as sp
# from scipy import stats
# import matplotlib.pyplot as plt
import numpy as np
import MDAnalysis as mda
import sys
from tqdm.contrib import itertools
from line_profiler import profile


# plt.style.use(
    # "https://raw.githubusercontent.com/sheamusfitz/mpl-styles/main/smallfigs.mplstyle")

# FOR DPPC
important_atoms = [
    ['O22', 'C21', 'O21'], 
    ['O32', 'C31', 'O31']
]
exclude_atoms = [
    ['C2', 'HS',          'O21', 'C21', 'O22',    'C22', 'H2S', 'H2R',    'C23', 'H3S', 'H3R'],
    ['C3', 'HX', 'HY',    'O31', 'C31', 'O32',    'C32', 'H2X', 'H2Y',    'C33', 'H3X', 'H3Y']
]

bohr_to_angstrom = 0.529177

@profile
def calc_efield(f, universe: mda.Universe, resid, cutoff):
    """
    this needs to return a 1x6 vector of the electric fields projected onto the C=O bond, at the atoms in "important_atoms"
    """
    out = np.zeros(6)

    universe.trajectory[f]

    dims = universe.dimensions[:3]

    for t, tail in enumerate(important_atoms):
        # print('tail:\t', tail)

        a0pos = universe.select_atoms(f'resid {resid} and name C{t+2}1').positions[0]
        a1pos = universe.select_atoms(f'resid {resid} and name O{t+2}2').positions[0]

        bond_vector = a1pos - a0pos
        unit_vector = bond_vector / np.linalg.norm(bond_vector)

        # solvent_atoms = universe.select_atoms(f'(around {cutoff} (resid {resid} and name {atom})) and not (resid {resid} and name {" ".join(exclude_atoms[t])})')
        solvent_atoms = universe.select_atoms(f'(around {cutoff} (resid {resid} and name {" ".join(tail)})) and not (resid {resid} and name {" ".join(exclude_atoms[t])})')
        solvent_positions = np.array(solvent_atoms.positions)
        charges = np.array(solvent_atoms.charges)
        # print('type(charges)', type(charges))

        loc_positions = np.array([
            universe.select_atoms(f'resid {resid} and name {atom}').positions for atom in tail
        ])[:, 0, :]

        
        #>>> r_is :: [(O22, C21, O21), n_solvent_atoms, (x,y,z)]
        r_is = solvent_positions[np.newaxis, :, :] - loc_positions[:, np.newaxis, :]
        
        r_is -= np.round(r_is / dims[np.newaxis, np.newaxis, :]) * dims[np.newaxis, np.newaxis, :] # PBC fix

        distances = np.linalg.norm(r_is, axis=2) / bohr_to_angstrom

        inv_distances = 1 / distances

        # print(inv_distances.shape)

        units = inv_distances[..., np.newaxis] * r_is

        # print('u0', units[0, 0])
        # print('r0', r_is[0, 0])
        # print('unit', unit_vector)
        # print(np.dot(units[0,0], unit_vector))
        # print(np.dot(units, unit_vector)[0,0])
        # print(charges.shape)
        efield_at_locatom = charges[np.newaxis, :] * np.dot(units, unit_vector) * inv_distances**2


        out[t*3:(t+1)*3] = np.sum(efield_at_locatom, axis=1)
        # raise Exception
        # for a, atom in enumerate(tail):
        #     # locatom = universe.select_atoms(f'resid {resid} and name {atom}')
        #     loc_position = locatom.positions[0]

        #     r_is = solvent_positions - loc_position

        #     r_is -= np.round(r_is / dims) * dims # PBC fix
            
        #     distances = np.linalg.norm(r_is, axis=1) / bohr_to_angstrom
        #     inv_distances = 1 / distances
        #     inv_distances_sq = inv_distances**2

        #     out[t*3+a] = np.sum(charges * np.dot(inv_distances[:, np.newaxis] * r_is, unit_vector) * inv_distances_sq)
    # print(out)
    return out



# @profile
def main():
    debug = bool(int(sys.argv[1]))

    system = sys.argv[2]
    if debug: print(">>== running in debug mode\n")
    universe = mda.Universe(
        f"/Users/shea/dcuments/research/psm-2dir/8nm/{system}/gromacs/step5_input.psf",
        f"/Users/shea/dcuments/research/psm-2dir/8nm/{system}/gromacs/production.xtc"
    )

    dppcs = universe.select_atoms('resname DPPC')
    dppc_resids = np.unique(dppcs.resids)
    num_resids = len(dppc_resids)
    if debug: 
        print(">>==", dppc_resids)
        print("len(universe.trajectory)", len(universe.trajectory))

    output_efield = np.zeros((len(universe.trajectory), num_resids*6))
    if debug: print('>>> l80'); 
    for f, (r, resid) in itertools.product(range(len(universe.trajectory)), enumerate(dppcs.residues.resids), total = len(universe.trajectory) * num_resids):
        # if f > 3:
        #     break
    # for f in range(len(universe.trajectory)):
    #     for r, resid in enumerate(dppcs.residues.resids):

        # calc_efield(f, universe, resid, 20)

        output_efield[f, r*6:(r+1)*6] = calc_efield(f, universe, resid, 20)
    np.save(f'{system}-efield.npy', output_efield)
            



if __name__ == '__main__':
    main()