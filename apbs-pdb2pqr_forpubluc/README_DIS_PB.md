# Running APBS Example

In this folder you can run **APBS** using the example setup from:
/home/ema/apbs-pdb2pqr_forpubluc/apbs/examples


### Files
- **`run.in`** – input file used to run APBS  
- **`fna.pqr`** – structure file representing an elongated coarse-grained DNA molecule  

### Output
Running APBS with this setup will generate the following files:

- **`efield_x.dx`**  
- **`efield_y.dx`**  
- **`efield_z.dx`**

These files contain the **derivatives of the electrostatic potential** along the respective coordinate axes.  

---

To run the calculation, simply execute APBS in this directory with the provided `run.in` file.

