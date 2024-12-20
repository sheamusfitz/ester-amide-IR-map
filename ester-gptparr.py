import MDAnalysis as mda
import numpy as np
import os
import time
from multiprocessing import Pool
from tqdm import tqdm
from functools import partial


def compute_electric_field(ts, universe, resid: int, atom0: str, atom1: str, cutoff, atoms_to_ignore: str = 'none'):
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
    
    for atom in solvent_atoms:
        r_i = atom.position - a0.positions[0]
        r_i -= np.round(r_i / universe.dimensions[:3]) * universe.dimensions[:3]  # PBC fix
        distance_i = np.linalg.norm(r_i)
        r_i_unit = r_i / distance_i
        q_i = atom.charge
        e_field_contribution = (q_i * r_i_unit) / (distance_i ** 2)
        e_field_at_a0 += e_field_contribution

    e_projection = np.dot(e_field_at_a0, unit_vector)
    return e_projection


def process_frame_chunk(chunk, universe_psf, universe_xtc, resids, atom0, atom1, cutoff, atoms_to_ignore):
    universe = mda.Universe(universe_psf, universe_xtc)
    chunk_results = np.zeros((len(resids), len(chunk)))
    for ts in chunk:
        t = ts
        for i, r in enumerate(resids):
            chunk_results[i, t - chunk[0]] = compute_electric_field(t, universe, r, atom0, atom1, cutoff, atoms_to_ignore)
    return chunk_results


def parallel_compute_electric_field(universe_psf, universe_xtc, resids, atom0, atom1, cutoff, atoms_to_ignore, n_frames_per_chunk, n_procs):
    universe = mda.Universe(universe_psf, universe_xtc)
    n_frames = len(universe.trajectory)
    
    # Create chunks with the appropriate number of frames
    chunks = [list(range(i, min(i + n_frames_per_chunk, n_frames))) for i in range(0, n_frames, n_frames_per_chunk)]
    pool_args = [(chunk, universe_psf, universe_xtc, resids, atom0, atom1, cutoff, atoms_to_ignore) for chunk in chunks]

    start_time = time.time()
    
    with Pool(processes=n_procs) as pool:
        process_chunk_with_args = partial(process_frame_chunk, universe_psf=universe_psf, universe_xtc=universe_xtc, resids=resids, atom0=atom0, atom1=atom1, cutoff=cutoff, atoms_to_ignore=atoms_to_ignore)
        
        # Use tqdm for progress tracking with average speed
        results = []
        with tqdm(total=len(pool_args), desc='Processing chunks', unit='chunk') as pbar:
            chunk_times = []
            for i, result in enumerate(pool.imap_unordered(process_chunk_with_args, chunks)):
                # Track time for each chunk
                chunk_start_time = time.time()
                chunk_results = result
                chunk_end_time = time.time()
                chunk_time = chunk_end_time - chunk_start_time
                
                chunk_times.append(chunk_time)
                results.append(chunk_results)
                
                # Calculate average processing speed
                elapsed_time = time.time() - start_time
                avg_speed = (i + 1) / elapsed_time if elapsed_time > 0 else 0
                
                pbar.set_postfix(avg_speed=f'{avg_speed:.4f} chunks/sec')
                pbar.update()

        end_time = time.time()
    
    # Concatenate results along the frame axis
    efield = np.hstack(results)
    
    # Calculate final average processing speed
    total_time = end_time - start_time
    avg_speed = len(pool_args) / total_time
    
    print(f"Total time: {total_time:.2f} seconds")
    print(f"Average processing speed: {avg_speed:.2f} chunks/second")
    
    return efield


if __name__ == '__main__':
    # Input data and parameters
    system = 'PSM-75'
    universe_psf = f'../../psm-2dir/8nm/{system}/gromacs/step5_input.psf'
    universe_xtc = f'../../psm-2dir/8nm/{system}/gromacs/production.xtc'
    resids = mda.Universe(universe_psf, universe_xtc).select_atoms('resname DPPC').residues.resids
    atom0 = 'C21'
    atom1 = 'O22'
    cutoff = 30
    atoms_to_ignore = f'resid {resids[0]} and name O21'
    n_procs = 6  # Adjust based on your system
    n_frames_per_chunk = 64  # Increase this value to reduce overhead

    should_run = True
    if os.path.isfile(f'../data/{system}-efield-parr.npy'):
        if os.path.getsize(f'../data/{system}-efield-parr.npy') > 0:
            should_run = False

    if should_run:
        tstart = time.time()
        efield = parallel_compute_electric_field(universe_psf, universe_xtc, resids, atom0, atom1, cutoff, atoms_to_ignore, n_frames_per_chunk, n_procs)
        
        with open(f'../data/{system}-efield-parr.npy', 'wb') as f:
            np.save(f, efield)
        
        tend = time.time()
        print(f"Total time: {tend - tstart:.2f} seconds")
