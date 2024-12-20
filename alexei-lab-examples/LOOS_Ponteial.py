import numpy as np 
import loos 
import loos.pyloos
import sys 



bohr_to_angstrom = 0.529177

hartree_to_MVcm = 5142.20652  # Convert Hartree atomic units to MV/cm

if __name__ in "__main__":
    # collect sys arguments
    coords = sys.argv[1]
    traj = sys.argv[2]
    # cutoff 
    cutoff = float(sys.argv[3])
    natom = sys.argv[4]
    catom = sys.argv[5]
    univers = sys.argv[6]

    # system intialzation 
    system = loos.createSystem(coords)
    traj = loos.pyloos.Trajectory(traj,system)
    # we can select atoms (will have to skip univers b/c it need to update per frame)
    catom = loos.selectAtoms(system,catom)
    natom = loos.selectAtoms(system,natom)
    uni = loos.selectAtoms(system,univers)
    # start trj analysis 
    for frame in traj:
        # select the atoms 
        box = uni.periodicBox()
        close = uni.within(30,catom,box)
        # n c postions 
    
        n = np.array([float(i) for i in natom.centerOfMass()])
        c = np.array([float(i) for i  in catom.centerOfMass()])
        rcn = c - n 
        # rcn N <-- C aka N-C 
        # convert to a unit vector 
        urcn = rcn / (np.linalg.norm(rcn))
        
        # we now find all of postions of the uni 
        xj = close.getCoords() # need to del this 
    
        # concevr these 
        rnj = xj - n  # del this too 
        rnj2 = np.linalg.norm(rnj,axis=1)
        #rnj2 = np.linalg.norm(rnj,axis=1)
        urnj = rnj / rnj2[:,None]
       

        # turn to borh 
        # borh distacne 
        borh = rnj2 / bohr_to_angstrom

        # charge 
        qi = np.array([float(i.charge()) for i in close])
        # start the math 
        efield = (urnj * qi[:,None]) / (borh[:,None]**2)
        total_field = efield.sum(axis=0)
        dot = np.dot(urcn,total_field)
        efeild_MV_CM  = dot * hartree_to_MVcm
        print(traj.index(),efeild_MV_CM)
        del xj 
        del rnj 
        del rnj2 
        del efield
        del qi 
        