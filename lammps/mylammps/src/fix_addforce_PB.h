/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#ifdef FIX_CLASS
// clang-format off
FixStyle(addforce_PB,FixAddForceForce);
// clang-format on
#else

#ifndef LMP_FIX_ADDFORCE_PB_H
#define LMP_FIX_ADDFORCE_PB_H

#include "fix.h"
#include<vector>
namespace LAMMPS_NS {

class FixAddForceForce : public Fix {
 public:
  FixAddForceForce(class LAMMPS *, int, char **);
  ~FixAddForceForce();
  int setmask() override;
  void init() override;
  void setup(int) override;
  void min_setup(int) override;
  void post_force(int) override;
  void post_force_respa(int, int, int) override;
  void min_post_force(int) override;
  double memory_usage() override;
  double compute_scalar() override;
  double compute_vector(int) override;

  enum { NONE, CONSTANT, EQUAL, ATOM };

 protected:
  double xvalue, yvalue, zvalue;
  int varflag, iregion;
  
  
  //double foriginal[4], foriginal_all[4];
  double fsum[4], fsum_all[4];
  int force_flag;
  int ilevel_respa;
  int maxatom, maxatom_energy;
  
  char *elecfield_y, *elecfield_z,*elecfield_x,*sx_Com, *sy_Com,*sz_Com,*sE_unit_kcal,*stype_Cm; //*sdelta *stype_Cl,*stype_Na;
  struct interpolar_input {int in_nx; int in_ny;int in_nz; double in_xmin; double in_xmax; double in_ymin; double in_ymax; double in_zmin; double in_zmax; double in_hx; double in_hy; double in_hz;};
  std::vector <interpolar_input> I;
  std::vector <interpolar_input> I_empty;
  std::vector < std::vector < std::vector<double> > > test_x;
  std::vector < std::vector < std::vector<double> > > test_y;
  std::vector < std::vector < std::vector<double> > > test_z;
  
  
  int x_Com, y_Com, z_Com,type_Cm; //type_Cl,type_Na
  double E_unit_kcal; //,delta
};

}    // namespace LAMMPS_NS

#endif
#endif

/* ERROR/WARNING messages:

E: Illegal ... command

Self-explanatory.  Check the input script syntax and compare to the
documentation for the command.  You can use -echo screen as a
command-line option when running LAMMPS to see the offending line.

E: Region ID for fix addforce does not exist

Self-explanatory.

E: Variable name for fix addforce does not exist

Self-explanatory.

E: Variable for fix addforce is invalid style

Self-explanatory.

E: Cannot use variable energy with constant force in fix addforce

This is because for constant force, LAMMPS can compute the change
in energy directly.

E: Must use variable energy with fix addforce

Must define an energy variable when applying a dynamic
force during minimization.

*/
