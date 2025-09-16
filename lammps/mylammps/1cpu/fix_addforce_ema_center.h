/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#ifdef FIX_CLASS
/*tako registrira Fix v LAMMPS; mogoce bi tudi v cpp moral biti S TEM NAJDEM FIX KO GA POKLICES IZ INPUT SCRIPTE LAMMPS;**/ 
FixStyle(addforce_ema_center,FixAddForceEmaCenter)

#else

#ifndef LMP_FIX_ADDFORCE_EMA_CENTER_H
#define LMP_FIX_ADDFORCE_EMA_CENTER_H

#include "fix.h"
#include<vector>
namespace LAMMPS_NS {

class FixAddForceEmaCenter : public Fix {
 public: //poves katere metode bos uporabil
  FixAddForceEmaCenter(class LAMMPS *, int, char **);
  ~FixAddForceEmaCenter();
  int setmask();
  void init();
  void setup_pre_force(int);
  //void setup(int);
  void min_setup(int);
  //void post_force(int);
  void pre_force(int);
  //void post_force_respa(int, int, int);
  //void min_post_force(int);
  //double compute_scalar();
  //double compute_vector(int);
  //double memory_usage();
  
  /*struct interpolar_input {const int in_nx; const int in_ny; const int in_nz; const int in_xmin; const int in_xmax; const int in_ymin; const int in_ymax; const float in_zmin; const int in_zmax; const float 	   in_hx; const float in_hy; const float in_hz;};
  std::vector <interpolar_input> I;
  std::vector < std::vector < std::vector<int> > > test_x;
  std::vector < std::vector < std::vector<int> > > test_y;
  std::vector < std::vector < std::vector<int> > > test_z;*/
 private: //ostali parametri = protected
  double xvalue,yvalue,zvalue;
  int varflag,iregion;
  char *xstr,*ystr,*zstr,*estr;
  char *idregion;
  char *edens_y, *edens_z,*edens_x,*scale_data_Cl,*scale_data_Na,*sx_Com, *sy_Com,*sz_Com,*sdelta,*ssmin,*ssmax,*sE_unit_kcal,*stype_Cl,*stype_Na,*stype_Cm,*dscale_data_Cl,*dscale_data_Na;
  int xvar,yvar,zvar,evar,xstyle,ystyle,zstyle,estyle;
  double foriginal[4],foriginal_all[4];
  int force_flag;
  int ilevel_respa;
  struct interpolar_input {int in_nx; int in_ny;int in_nz; double in_xmin; double in_xmax; double in_ymin; double in_ymax; double in_zmin; double in_zmax; double in_hx; double in_hy; double in_hz;};
  std::vector <interpolar_input> I;
  std::vector <interpolar_input> I_empty;
  std::vector <double> data_sigma_Na;
  std::vector <double> data_mean_Na;
  std::vector <double> data_sigma_Cl;
  std::vector <double> data_mean_Cl;
  std::vector <double> data_min_sym_Cl;
  std::vector <double> data_max_sym_Cl;
  std::vector <double> data_min_sym_Na;
  std::vector <double> data_max_sym_Na;
  std::vector <double> data_dsigma_Na;
  std::vector <double> data_dmean_Na;
  std::vector <double> data_dsigma_Cl;
  std::vector <double> data_dmean_Cl;
  std::vector <double> data_dmin_sym_Cl;
  std::vector <double> data_dmax_sym_Cl;
  std::vector <double> data_dmin_sym_Na;
  std::vector <double> data_dmax_sym_Na;
  std::vector < std::vector < std::vector<double> > > test_x;
  std::vector < std::vector < std::vector<double> > > test_y;
  std::vector < std::vector < std::vector<double> > > test_z;
  int maxatom;
  double **sforce;
  // int ema_var;
  int ema_var2, x_Com, y_Com, z_Com,type_Cl,type_Na,type_Cm;
  double smin, smax, E_unit_kcal,delta;
   void efield_calc();

};

}

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
