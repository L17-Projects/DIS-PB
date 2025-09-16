// clang-format off
/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#include "fix_addforce_PB.h"

#include "atom.h"
//#include "atom_masks.h"
#include "comm.h"
#include "domain.h"
#include "error.h"
#include "force.h"
#include "input.h"
#include "memory.h"
#include "modify.h"
#include "pair.h"
#include "region.h"
#include "respa.h"
#include "update.h"
#include "variable.h"


#include <iostream>
using namespace std;
#include <cstring>
#include <math.h>
#include <stdlib.h> 
#include<fstream>
#include<istream>
#include<vector>
#include<sstream>
#include <typeinfo>
#include <assert.h> 
#include <iomanip>
#include <numeric>


using namespace LAMMPS_NS;
using namespace FixConst;

//enum{NONE,CONSTANT,EQUAL,ATOM};

/* ---------------------------------------------------------------------- */

FixAddForceForce::FixAddForceForce(LAMMPS *lmp, int narg, char **arg) :
  Fix(lmp, narg, arg)
{
  if (narg < 6) error->all(FLERR,"Illegal fix addforce command");

  dynamic_group_allow = 1;
  scalar_flag = 1;
  vector_flag = 1;
  size_vector = 3;
  global_freq = 1;
  extscalar = 1;
  extvector = 1;
  energy_global_flag = 1;
  virial_global_flag = virial_peratom_flag = 1;
  respa_level_support = 1;
  ilevel_respa = 0;

  
  
  elecfield_y = arg[5];
  elecfield_x = arg[4];
  elecfield_z = arg[6];
  sx_Com = arg[7];
  sy_Com = arg[8];
  sz_Com = arg[9];
  
  sE_unit_kcal = arg[10];
 
  
 

  force_flag = 0;
 
  fsum[0] = fsum[1] = fsum[2] = fsum[3] = 0.0;
  maxatom = atom->nmax; // v org  maxatom = 1;
  maxatom_energy = 0;

}

/* ---------------------------------------------------------------------- */

FixAddForceForce::~FixAddForceForce()
{ if (copymode) return;
 
}

/* ---------------------------------------------------------------------- */

int FixAddForceForce::setmask()
{
  
  int mask = 0;
  mask |= POST_FORCE;
  mask |= POST_FORCE_RESPA;
  mask |= MIN_POST_FORCE;
  return mask;
}

/* ---------------------------------------------------------------------- */

void FixAddForceForce::init()
{
  // check variables

  

  if (utils::strmatch(update->integrate_style,"^respa")) {
    ilevel_respa = ((Respa *) update->integrate)->nlevels-1;
    if (respa_level >= 0) ilevel_respa = MIN(respa_level,ilevel_respa);
  }
  test_x.clear();
  std::vector < std::vector < std::vector<double> > >().swap(test_x);
  test_y.clear();
  std::vector < std::vector < std::vector<double> > >().swap(test_y);
  test_z.clear();
  std::vector < std::vector < std::vector<double> > >().swap(test_z);
  I.clear();
  vector <interpolar_input>().swap(I);
  vector <double> data_x;
  data_x.clear();
  vector<double>().swap(data_x);
  vector <double> data_y;
  data_y.clear();
  vector<double>().swap(data_y);
  vector <double> data_z;
  data_z.clear();
  vector<double>().swap(data_z);
  string snx, sxmin, shx;
  string sny, symin,shy;
  string snz, szmin,shz;
  string sefield_x;
  double efield_x;
  string sefield_y;
  double efield_y;
  string sefield_z;
  double efield_z;
  
  ifstream file_x(elecfield_x);
   
    string line_x; 
    while (getline(file_x, line_x)){
        vector <string> v;
        v.clear();
        istringstream ss(line_x);
        string word;
        while (ss >> word) {
        v.push_back(word);
        }
        if (v[0] == "#"){continue;}
        
         if (v[0] == "object" && v[3] == "gridpositions"){
             snx = v[5];
             sny = v[6];
             snz = v[7];

        
        }
        if (v[0] == "origin"){
            sxmin = v[1];
            symin = v[2];
            szmin = v[3];
            
        }
        if (v[0] == "delta"){
            if (v[1] != "0.000000e+00"){
            shx = v[1];}
            if (v[2] != "0.000000e+00"){
            shy = v[2];}
            if (v[3] != "0.000000e+00"){
                shz = v[3];
                }
        }
       if (v[0] != "object" &&  v[0] != "origin" && v[0] != "delta"&& v[0] != "attribute"&& v[0] != "component"){
           for (int index = 0; index < v.size(); ++index) {
               
               sefield_x = v[index];
               stringstream gsefield(sefield_x);
               efield_x = 0.0;
               gsefield >> efield_x;
               data_x.push_back(efield_x);}} 
    }
    file_x.close();
    ifstream file_y(elecfield_y);
    string line_y; 
    while (getline(file_y, line_y)){
        vector <string> v_y;
        v_y.clear();
        istringstream ss(line_y);
        string word_y;
        while (ss >> word_y) {
        v_y.push_back(word_y);
        }
        if (v_y[0] == "#"){continue;}
        if (v_y[0] != "object" &&  v_y[0] != "origin" && v_y[0] != "delta"&& v_y[0] != "attribute"&& v_y[0] != "component"){
        for (int index = 0; index < v_y.size(); ++index) {
             
               sefield_y = v_y[index];
               stringstream gsefield(sefield_y);
               efield_y = 0.0;
               gsefield >> efield_y;
               data_y.push_back(efield_y);}} 
    }
    file_y.close();
  
   
    ifstream file_z(elecfield_z);
    string line_z; 
    while (getline(file_z, line_z)){

        vector <string> v_z;
        v_z.clear();
        istringstream ss(line_z);
        string word_z;
        while (ss >> word_z) {
        v_z.push_back(word_z);
        }
        if (v_z[0] == "#"){continue;}
         if (v_z[0] != "object" &&  v_z[0] != "origin" && v_z[0] != "delta"&&  v_z[0] != "attribute" && v_z[0] != "component"){
           for (int index = 0; index < v_z.size(); ++index) {
              
               sefield_z = v_z[index];
               stringstream gsefield(sefield_z);
               efield_z = 0.0;
               gsefield >> efield_z;
               data_z.push_back(efield_z);}} 
    }

    file_z.close();


    stringstream gsnx(snx);
    stringstream gsny(sny);
    stringstream gsnz(snz);
    stringstream gsxmin(sxmin);
    stringstream gsymin(symin);
    stringstream gszmin(szmin);
    stringstream gshx(shx);
    stringstream gshy(shy);
    stringstream gshz(shz);
    int nx = 0;
    int ny=0;
    int nz=0;
    double xmin=0.0;
    double ymin=0.0;
    double zmin=0.0;
    double hx = 0.0;
    double hy = 0.0;
    double hz = 0.0;
    gsnx >> nx;
    gsny >> ny;
    gsnz >> nz;
    gsxmin >> xmin;
    gsymin >> ymin;
    gszmin >> zmin;
    gshx >> hx;
    gshy >> hy;
    gshz >> hz;
    double xmax=xmin+(double(nx-1))*hx;
    double ymax=ymin+(double(ny-1))*hy;
    double zmax=zmin+(double(nz-1))*hz;
    I.push_back({nx, ny, nz, xmin, xmax,ymin, ymax, zmin, zmax, hx, hy, hz});
    cout <<"xmin "<<I[0].in_xmin<<"xmax "<<I[0].in_xmax<<"ymin "<<I[0].in_ymin<<"ymax "<<I[0].in_ymax<<"zmin "<<I[0].in_zmin<<"zmax "<<I[0].in_zmax<<endl;
    cout<<"hx "<<I[0].in_hx<<"hy "<<I[0].in_hy<<"hz "<<I[0].in_hz<<"nx "<<I[0].in_nx<<"ny "<<I[0].in_ny<<"nz "<<I[0].in_nz<<endl;
    int countLines= 0;
    cout <<"here"<<endl;
    
   
    for (int i = 0; i < nx; ++i) {
        test_x.push_back(vector<vector<double>>());
        test_y.push_back(vector<vector<double>>());
        test_z.push_back(vector<vector<double>>());
       
        for (int j = 0; j < ny; ++j) {
            test_x[i].push_back(vector<double>());
            test_y[i].push_back(vector<double>());
            test_z[i].push_back(vector<double>());
           
            for (int k = 0; k <nz; ++k) {
                countLines++;
                test_x[i][j].push_back(data_x[countLines-1]);
                test_y[i][j].push_back(data_y[countLines-1]);
                test_z[i][j].push_back(data_z[countLines-1]);
               
            }
        }
    }

   
    
    stringstream gsx_Com(sx_Com);
    stringstream gsy_Com(sy_Com);
    stringstream gsz_Com(sz_Com);
  
    
    stringstream gsE_unit_kcal(sE_unit_kcal);
 
   
    int x_Com_val = 0;
    int y_Com_val=0;
    int z_Com_val=0;

   
    double E_unit_kcal_val= 0.0;
  
    int type_Cm_val=0;
 
    gsx_Com >> x_Com_val;
    gsy_Com >> y_Com_val;
    gsz_Com >> z_Com_val;

    
    gsE_unit_kcal >> E_unit_kcal_val;
  
   
    x_Com = x_Com_val;
    y_Com = y_Com_val;
    z_Com = z_Com_val;
   
    E_unit_kcal = E_unit_kcal_val;
  
    type_Cm = type_Cm_val;
    cout<<" x_Com "<<x_Com<<" y_Com "<<y_Com<<" z_Com "<<z_Com<<" Eunit_kcal "<<E_unit_kcal<<" type Cm "<<type_Cm<<endl;
}

/* ---------------------------------------------------------------------- */

void FixAddForceForce::setup(int vflag)
{
  if (utils::strmatch(update->integrate_style, "^respa")) {
    auto respa = dynamic_cast<Respa *>(update->integrate);
    respa->copy_flevel_f(ilevel_respa);
    post_force_respa(vflag, ilevel_respa, 0);
    respa->copy_f_flevel(ilevel_respa);
  } else {
    post_force(vflag);
  }
}

/* ---------------------------------------------------------------------- */

void FixAddForceForce::min_setup(int vflag)
{
  post_force(vflag);
}

/* ---------------------------------------------------------------------- */

void FixAddForceForce::post_force(int vflag)
{
  
  double **f = atom->f;
  int *mask = atom->mask;
  imageint *image = atom->image;
  tagint *id = atom->tag;
  double *q = atom->q;
  int *type = atom->type;
  int nlocal = atom->nlocal;

  
  
  
  
  double x_Cm, y_Cm, z_Cm;
  double x_delta, y_delta, z_delta;
  double x_pos,y_pos,z_pos;
  double x_pos_move, y_pos_move, z_pos_move;

  int id_atom;


  v_init(vflag);
  maxatom = atom->nmax;

  

  
  fsum[0] = fsum[1] = fsum[2] = fsum[3] = 0.0;
  force_flag = 0;
    
  
  double **x = atom->x;
  double v[6],unwrap[3];

  
 

    for (int index = 0; index < nlocal; index++) {
        if (mask[index] & groupbit) {
       
        
        
        x_pos = (x[index][0]); //pos v angrstremih
        y_pos = (x[index][1]);
        z_pos = (x[index][2]);
        
    
    
    
        id_atom = id[index];
        
        double charg = q[index];
 
        
        
        
        
        vector <double> x_baze;
        vector <double> z_baze;
        vector <double> y_baze;
        
        
        vector <double> points_x;
        vector <double> points_y;
        vector <double> points_z;
        
      
        
        if ((x_pos > (I[0].in_xmax))|| (x_pos < (I[0].in_xmin))) {cout << "x out of range "<<x_pos<< "  "<< I[0].in_xmax-x_Com<<endl; exit(EXIT_FAILURE);}
        if ((y_pos > (I[0].in_ymax) )|| (y_pos < (I[0].in_ymin))) {cout << "y out of range "<< y_pos<<endl; exit(EXIT_FAILURE);}
        if ((z_pos > (I[0].in_zmax))|| (z_pos < (I[0].in_zmin))) {cout << "z out of range "<< z_pos<<" min "<<(I[0].in_zmin-z_Com)<<" max "<<(I[0].in_zmax -z_Com)<<endl; exit(EXIT_FAILURE);}
        
        vector <double> list;
       
        size_t ihi, jhi, khi, ilo, jlo, klo;    
      
        double ifloat, jfloat, kfloat;
        double u_x = 0.0;
        double u_y = 0.0;
        double u_z = 0.0;
        
     
        
        ifloat = (x_pos-(I[0].in_xmin))/I[0].in_hx;
        jfloat = (y_pos-(I[0].in_ymin))/I[0].in_hy;
        kfloat = (z_pos-(I[0].in_zmin))/I[0].in_hz;
        
     
        
        ihi = (int)ceil(ifloat);
        ilo = (int)floor(ifloat);
        jhi = (int)ceil(jfloat);
        jlo = (int)floor(jfloat);
        khi = (int)ceil(kfloat);
        klo = (int)floor(kfloat);
        
        int Vgrid_digits = 9; //tudi v APBS toliko 6
        double Vcompare = pow(10,-1*(Vgrid_digits - 2));
        if ((x_pos > ((I[0].in_xmin))) && (y_pos > ((I[0].in_ymin))) && (z_pos > ((I[0].in_zmin)))){
        if ((ihi<I[0].in_nx) && (jhi<I[0].in_ny) && (khi<I[0].in_nz))// && (ilo>=0.0) && (jlo>=0.0) && (klo>=0.0)) //&& ((ihi != ilo) && (jhi != jlo) && (khi != klo))){
        {
        double dx = (ifloat - double(ilo));
        double dy = (jfloat - double(jlo));
        double dz = (kfloat - double(klo));
        int a1,a2,a3,a4,a5,a6,a7,a8;
        int b1,b2,b3,b4,b5,b6,b7,b8;
        int c1,c2,c3,c4,c5,c6,c7,c8;
        
        a1 = a2 = a3 = a4 = ihi;
        a5 = a6=a7=a8 = ilo;
        b1 = b3 = b5 = b7 = jhi;
        b2 = b4=b6=b8 = jlo;
        c1=c2=c5=c6= khi;
        c3=c4=c7=c8 = klo;
        
        u_x =(dx      *dy      *dz      *test_x[a1][b1][c1]
        + dx      *(1.0-dy)*dz      *test_x[a2][b2][c2]
        + dx      *dy      *(1.0-dz)*test_x[a3][b3][c3]
        + dx      *(1.0-dy)*(1.0-dz)*test_x[a4][b4][c4]
        + (1.0-dx)*dy      *dz      *test_x[a5][b5][c5]
        + (1.0-dx)*(1.0-dy)*dz      *test_x[a6][b6][c6]
        + (1.0-dx)*dy      *(1.0-dz)*test_x[a7][b7][c7]
        + (1.0-dx)*(1.0-dy)*(1.0-dz)*test_x[a8][b8][c8]);
        
        u_y =(dx      *dy      *dz      *test_y[a1][b1][c1]
        + dx      *(1.0-dy)*dz      *test_y[a2][b2][c2]
        + dx      *dy      *(1.0-dz)*test_y[a3][b3][c3]
        + dx      *(1.0-dy)*(1.0-dz)*test_y[a4][b4][c4]
        + (1.0-dx)*dy      *dz      *test_y[a5][b5][c5]
        + (1.0-dx)*(1.0-dy)*dz      *test_y[a6][b6][c6]
        + (1.0-dx)*dy      *(1.0-dz)*test_y[a7][b7][c7]
        + (1.0-dx)*(1.0-dy)*(1.0-dz)*test_y[a8][b8][c8]);
        
        u_z =(dx      *dy      *dz      *test_z[a1][b1][c1]
        + dx      *(1.0-dy)*dz      *test_z[a2][b2][c2]
        + dx      *dy      *(1.0-dz)*test_z[a3][b3][c3]
        + dx      *(1.0-dy)*(1.0-dz)*test_z[a4][b4][c4]
        + (1.0-dx)*dy      *dz      *test_z[a5][b5][c5]
        + (1.0-dx)*(1.0-dy)*dz      *test_z[a6][b6][c6]
        + (1.0-dx)*dy      *(1.0-dz)*test_z[a7][b7][c7]
        + (1.0-dx)*(1.0-dy)*(1.0-dz)*test_z[a8][b8][c8]); 
            
        }}
        
        
        else{exit(EXIT_FAILURE);
           }
       
        double x_efield = -u_x *E_unit_kcal; // //F[eV/A] =- E[V/A]*q - ker iz APBS je samo gradient pot
        double y_efield = -u_y *E_unit_kcal; //
        double z_efield = -u_z *E_unit_kcal;
        
   
        

        f[index][0] += x_efield * charg;
        f[index][1] += y_efield * charg;
        f[index][2] += z_efield * charg;
      
      
        domain->unmap(x[index],image[index],unwrap);
      
       
        fsum[0] -= (x_efield * charg)*unwrap[0] + (y_efield * charg)*unwrap[1] + (z_efield * charg)*unwrap[2]; 
        fsum[1] += x_efield * charg; 
        fsum[2] += y_efield * charg;
        fsum[3] += z_efield * charg;
        
      
        if (evflag) {
           
          v[0] = (x_efield * charg)*unwrap[0];
          v[1] = (y_efield * charg)*unwrap[1];
          v[2] = (z_efield * charg)*unwrap[2];
          v[3] = (x_efield * charg)*unwrap[1];
          v[4] = (x_efield * charg)*unwrap[2];
          v[5] = (y_efield * charg)*unwrap[2];
         
          v_tally(index,v);
        }
      }
    }
}


/* ---------------------------------------------------------------------- */

void FixAddForceForce::post_force_respa(int vflag, int ilevel, int /*iloop*/)
{
  if (ilevel == ilevel_respa) post_force(vflag);
}

/* ---------------------------------------------------------------------- */

void FixAddForceForce::min_post_force(int vflag)
{
  post_force(vflag);
}

/* ----------------------------------------------------------------------
   potential energy of added force
------------------------------------------------------------------------- */

double FixAddForceForce::compute_scalar()
{
  // only sum across procs one time

  if (force_flag == 0) {
    //MPI_Allreduce(foriginal,foriginal_all,4,MPI_DOUBLE,MPI_SUM,world);
    MPI_Allreduce(fsum,fsum_all,4,MPI_DOUBLE,MPI_SUM,world);
    force_flag = 1;
  }
  return fsum_all[0];
}

/* ----------------------------------------------------------------------
   return components of total force on fix group before force was changed
------------------------------------------------------------------------- */

double FixAddForceForce::compute_vector(int n)
{
  // only sum across procs one time

  if (force_flag == 0) {
    //MPI_Allreduce(foriginal,foriginal_all,4,MPI_DOUBLE,MPI_SUM,world);
    MPI_Allreduce(fsum,fsum_all,4,MPI_DOUBLE,MPI_SUM,world);
    force_flag = 1;
  }
  return fsum_all[n+1];
}

/* ----------------------------------------------------------------------
   memory usage of local atom-based array
------------------------------------------------------------------------- */

double FixAddForceForce::memory_usage()
{
  double bytes = 0.0;
  //if (varflag == ATOM) bytes = maxatom*4 * sizeof(double);
  bytes = atom->nmax * 4 * sizeof(double);
  return bytes;
}
