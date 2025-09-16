# DIS-PB MD Simulation

This example demonstrates how to perform **molecular dynamics (MD) simulations** of a **fixated coarse-grained CG-DNA molecule** with surrounding ions, using the **DIS-PB model**.

Before running the simulation, you should also:  
- Create a **conda environment** with all appropriate PyTorch requirements, as described in the [Allegro repository](https://github.com/mir-group/allegro), to properly run `pair_allegro`.  
- Build the LAMMPS code in a **new build directory**, making sure it links against the Allegro environment and includes the custom `fix_addforce_PB` module.

### Steps

1. **Prepare LAMMPS input**  
   - Create a `in.run` file in:  
     ```
     mylammps/examples/allegro/
     ```  
   - This input script should include the custom fix module:  
     ```
     fix_addforce_PB
     ```  
   - Ensure the files `fix_addforce_PB.cpp` and `fix_addforce_PB.h` are located in the LAMMPS `src/` directory and compiled into your LAMMPS build.

2. **Deploy the trained Allegro model**  
   - Copy the trained Allegro potential file (`ff.pth`) from the training step into:  
     ```
     mylammps/examples/allegro/cg_dna/ff/ff.pth
     ```  
   - This file contains the **potential of mean force** used in the MD simulation.

3. **Import electrostatic field files**  
   - Copy the following files, previously calculated with APBS, into:  
     ```
     mylammps/examples/allegro/cg_dna/
     ```  
     - `efield_x.dx`  
     - `efield_y.dx`  
     - `efield_z.dx`  
   - These fields are required to add back the PB forces that were subtracted during the preparation of the training data.  

---

At the end of this workflow you will be able to run **LAMMPS MD simulations** of a fixated CG DNA molecule with ions, using the **DIS-PB model** coupled with APBS-derived electrostatic fields.  

To start the simulation, run from the `mylammps/examples/allegro/` directory:  

```
lmp -in in.run

