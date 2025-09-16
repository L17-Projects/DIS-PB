# Data Preparation

This example demonstrates how to prepare training data for coarse-grained DNA (CG_DNA) in 1 mol/L NaCl solution.

### Requirements
1. An **all-atom MD LAMMPS trajectory**.  
2. The **electrostatic field files** (`efield_x.dx`, `efield_y.dx`, `efield_z.dx`) previously calculated with **APBS**.  

### Steps

1. **Calculate PB forces**  
   - Import the previously computed APBS field files:  
     - `efield_x.dx`  
     - `efield_y.dx`  
     - `efield_z.dx`  
   - Take the all-atom MD trajectory and provide it as `input_file` to `PB_forces.cpp`.  
   - Compile and execute `PB_forces.cpp`.  
   - This calculates **Poisson–Boltzmann (PB) forces** for the ions at their positions in the all-atom trajectory.  
   - The PB forces are saved alongside the original MD forces inside the trajectory file.  

2. **Prepare Allegro training data**  
   - Run the script `write-xyz-dif-traypb.py`.  
   - This generates the `input.data` file needed for **Allegro training**, containing CG_DNA atoms and ions.  

---

At the end of this workflow you will have a properly formatted `input.data` file for training your coarse-grained DNA model with Allegro.

