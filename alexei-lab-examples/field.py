import MDAnalysis
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from functools import partial
from scipy.stats import norm

# Constants for the conversion
bohr_to_angstrom = 0.529177
hartree_to_MVcm = 5142.20652  # Convert Hartree atomic units to MV/cm

def compute_electric_field(ts, universe, n3_atom, c13_atom, solvent_atoms):
    e_field_at_n3 = np.zeros(3)
    universe.trajectory[ts]
    cn_vector = c13_atom.positions[0] - n3_atom.positions[0]
    cn_unit_vector = cn_vector / np.linalg.norm(cn_vector)

    for atom in solvent_atoms:
        r_jn = atom.position - n3_atom.positions[0]
        r_jn -= np.round(r_jn / universe.dimensions[:3]) * universe.dimensions[:3]  # PBC fix
        distance_jn = np.linalg.norm(r_jn)
        r_jn_unit = r_jn / distance_jn
        distance_jn_bohr = distance_jn / bohr_to_angstrom
        q_j = atom.charge
        e_field_contribution = (q_j * r_jn_unit) / (distance_jn_bohr ** 2)
        e_field_at_n3 += e_field_contribution

    e_projection = np.dot(e_field_at_n3, cn_unit_vector)
    return e_projection * hartree_to_MVcm

def main():
    universe = MDAnalysis.Universe('md.tpr', 'md.xtc')
    n3_atom = universe.select_atoms('resname BOG and name C13')
    c13_atom = universe.select_atoms('resname BOG and name N3')
    solvent_atoms = universe.select_atoms('resname MOL')

    electric_field_values = [
        compute_electric_field(ts, universe, n3_atom, c13_atom, solvent_atoms)
        for ts in range(len(universe.trajectory))
    ]

    electric_field_MVcm = np.array(electric_field_values)
    np.savetxt('electric_field_data.dat', electric_field_MVcm, header='Electric Field (MV/cm)')
    print("The average electric field along the CN bond is {:.2f} MV/cm.".format(np.mean(electric_field_MVcm)))

    plot_results(electric_field_MVcm)

def plot_results(electric_field_MVcm):
    # Plotting code
    plt.figure(figsize=(12, 6))
    sns.histplot(electric_field_MVcm, bins=50, kde=True, color='magenta', stat='probability', line_kws={'color': 'black'})
    plt.xlabel('Electric Field (MV/cm)', fontsize=14)
    plt.ylabel('Probability Density', fontsize=14)
    plt.title('Probability Distribution of Electric Field along CN bond', fontsize=16)
    plt.xlim([-100, 50])
    plt.ylim([0, 0.08])
    plt.tick_params(axis='both', which='major', labelsize=12, length=8, width=1.5)
    plt.savefig('electric_field_distribution.png')

    # Additional plots as required
    # Repeat the pattern for other plots

if __name__ == "__main__":
    main()

