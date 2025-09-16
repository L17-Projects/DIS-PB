/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://lammps.sandia.gov/, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#include "fix_addforce_ema_center.h"
/*#include "math_extra.h"*/
#include "atom.h"
#include "atom_masks.h"
#include "domain.h"
#include "error.h"
#include "input.h"
#include "memory.h"
#include "modify.h"
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

//template <typename T> std::string type_name();
using namespace LAMMPS_NS;
using namespace FixConst;
//class je FIX vse tukaj je objecti tega classa

/*eflag != 0 means: compute energy contributions in this step
vflag != 0 means: compute virial contributions in this step

the exact value indicates whether per atom or total contributions
are supposed to be computed.*/


enum{NONE,CONSTANT,EQUAL,ATOM};


/* ---------------------------------------------------------------------- */
/*vsi fixi so razviti iz class Fix, in imajo constructor s 'podpisom'):*/
//constructor classa : nima return in je vedno public
 /*implementiras constructor; narg-stevilo argumentov; arg-list argumentov*/

FixAddForceEmaCenter::FixAddForceEmaCenter(LAMMPS *lmp, int narg, char **arg) :
  Fix(lmp, narg, arg)
  //xstr(nullptr), ystr(nullptr), zstr(nullptr), estr(nullptr), idregion(nullptr), sforce(nullptr), edens_x(nullptr)
{
  if (narg < 7) error->all(FLERR,"Illegal fix addforce command");

  dynamic_group_allow = 1;
  scalar_flag = 1;
  vector_flag = 1;
  size_vector = 3;
  global_freq = 1;
  extscalar = 1;
  extvector = 1;
  respa_level_support = 1;
  //ilevel_respa = 0;
  //virial_flag = 1;

  //xstr = ystr = zstr = edens_x= nullptr;
  

  //nothing = arg[3];
  edens_y = arg[5];
  edens_x = arg[4];
  edens_z = arg[6];
  scale_data_Na = arg[7];
  scale_data_Cl = arg[8];
  dscale_data_Na = arg[18];
  dscale_data_Cl = arg[19];
  sx_Com = arg[9];
  sy_Com = arg[10];
  sz_Com = arg[11];
  sdelta = arg[12];
  ssmin = arg[13];
  ssmax = arg[14];
  sE_unit_kcal = arg[15];
  stype_Cl = arg[16];
  stype_Na = arg[17];
  stype_Cm = arg[18];
  comm_forward = 1;
  /*if (strstr(arg[3],"v_") == arg[3]) {
    int n = strlen(&arg[3][2]) + 1;
    xstr = new char[n];
    strcpy(xstr,&arg[3][2]);
  } else {
    xvalue = utils::numeric(FLERR,arg[3],false,lmp);
    xstyle = CONSTANT;
  }
  if (strstr(arg[4],"v_") == arg[4]) {
    int n = strlen(&arg[4][2]) + 1;
    ystr = new char[n];
    strcpy(ystr,&arg[4][2]);
  } else {
    yvalue = utils::numeric(FLERR,arg[4],false,lmp);
    ystyle = CONSTANT;
  }
  if (strstr(arg[5],"v_") == arg[5]) {
    int n = strlen(&arg[5][2]) + 1;
    zstr = new char[n];
    strcpy(zstr,&arg[5][2]);
  } else {
    zvalue = utils::numeric(FLERR,arg[5],false,lmp);
    zstyle = CONSTANT;
  }*/

  // optional args

  /*nevery = 1;
  iregion = -1;

  int iarg = 6;
  while (iarg < narg) {
    if (strcmp(arg[iarg],"every") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal fix addforce command");
      nevery = atoi(arg[iarg+1]);
      if (nevery <= 0) error->all(FLERR,"Illegal fix addforce command");
      iarg += 2;
    } else if (strcmp(arg[iarg],"region") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal fix addforce command");
      iregion = domain->find_region(arg[iarg+1]);
      if (iregion == -1)
        error->all(FLERR,"Region ID for fix addforce does not exist");
      int n = strlen(arg[iarg+1]) + 1;
      idregion = new char[n];
      strcpy(idregion,arg[iarg+1]);
      iarg += 2;
    } else if (strcmp(arg[iarg],"energy") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal fix addforce command");
      if (strstr(arg[iarg+1],"v_") == arg[iarg+1]) {
        int n = strlen(&arg[iarg+1][2]) + 1;
        estr = new char[n];
        strcpy(estr,&arg[iarg+1][2]);
      } else error->all(FLERR,"Illegal fix addforce command");
      iarg += 2;
    } else error->all(FLERR,"Illegal fix addforce command");
  }*/
  //int ema_var = 2;
 //cout <<ema_var<<"ema_var_construct"<<endl;
  //force_flag = 0;
  //foriginal[0] = foriginal[1] = foriginal[2] = foriginal[3] = 0.0;

  maxatom = 1;
  comm_forward = 9;
  //memory->create(sforce,maxatom,4,"addforce:sforce");
}

/* ---------------------------------------------------------------------- */
//deconstructor sprotsi spomin
FixAddForceEmaCenter::~FixAddForceEmaCenter()
{
 // delete [] xstr;
  //delete [] ystr;
 // delete [] zstr;
  //delete [] estr;
  //delete [] idregion;
  //delete [] edens_z;
  //delete [] edens_x;
  //delete [] edens_y;
  //memory->destroy(sforce);
}

/* ---------------------------------------------------------------------- */
/*metoda setmask() doloci katera metoda bo poklicana med izvedbo. Izbiras med: initial_integrate, post_integrate, pre_exchange, pre_neighbor, pre_force, post_force, final_integrate, end_of_step; tako gredo po vrsti v verle.cpp. MORAS VEDETI KDAJ!*/
int FixAddForceEmaCenter::setmask() //to je metoda-funkcija classa FIxAddForceEma klicana zunaj tega classa; determines when the fix is called during the timestep
{//cout << edens_x<<" begining setmask1"<<endl;
  //datamask_read = datamask_modify = 0;

  int mask = 0;
  //mask |= FixConst::POST_FORCE;
  //mask |= FixConst::PRE_FORCE;//metoda post force()
  mask |= PRE_FORCE;
  //mask |= THERMO_ENERGY;
  mask |= POST_FORCE_RESPA; //če je update metoda RESPA in ne verlet
  mask |= MIN_POST_FORCE; //to je post force metoda za minimizacijo

  /*cout << edens_x<<" end setmask1"<<endl;
  cout << edens_y<<" end setmask1"<<endl;
  cout << edens_z<<" end setmask1"<<endl;*/
  return mask;
  
}

/* ---------------------------------------------------------------------- */

//funckija classa FixAddForceEma za initializacijo spremenjlivk; initialization before a run
void FixAddForceEmaCenter::init()
 // tu dodaj! pred vsakim run pokličejo
{   neighbor->add_request(this, NeighConst::REQ_FULL);
    
    /*cout << edens_x<<  " begining init"<<endl;
    cout << edens_y<<  " begining init"<<endl;
    cout << edens_z<<  " begining init"<<endl;*/
  // check variables

 /* if (xstr) {
    xvar = input->variable->find(xstr);
    if (xvar < 0)
      error->all(FLERR,"Variable name for fix addforce does not exist");
    if (input->variable->equalstyle(xvar)) xstyle = EQUAL;
    else if (input->variable->atomstyle(xvar)) xstyle = ATOM;
    else error->all(FLERR,"Variable for fix addforce is invalid style");
  }
  if (ystr) {
    yvar = input->variable->find(ystr);
    if (yvar < 0)
      error->all(FLERR,"Variable name for fix addforce does not exist");
    if (input->variable->equalstyle(yvar)) ystyle = EQUAL;
    else if (input->variable->atomstyle(yvar)) ystyle = ATOM;
    else error->all(FLERR,"Variable for fix addforce is invalid style");
  }
  if (zstr) {
    zvar = input->variable->find(zstr);
    if (zvar < 0)
      error->all(FLERR,"Variable name for fix addforce does not exist");
    if (input->variable->equalstyle(zvar)) zstyle = EQUAL;
    else if (input->variable->atomstyle(zvar)) zstyle = ATOM;
    else error->all(FLERR,"Variable for fix addforce is invalid style");
  }
  if (estr) {
    evar = input->variable->find(estr);
    if (evar < 0)
      error->all(FLERR,"Variable name for fix addforce does not exist");
    if (input->variable->atomstyle(evar)) estyle = ATOM;
    else error->all(FLERR,"Variable for fix addforce is invalid style");
  } else estyle = NONE;
*/
  // set index and check validity of region

  /*if (iregion >= 0) {
    iregion = domain->find_region(idregion);
    if (iregion == -1)
      error->all(FLERR,"Region ID for fix addforce does not exist");
  }*/

 /* if (xstyle == ATOM || ystyle == ATOM || zstyle == ATOM)
    varflag = ATOM;
  else if (xstyle == EQUAL || ystyle == EQUAL || zstyle == EQUAL)
    varflag = EQUAL;
  else varflag = CONSTANT;

  if (varflag == CONSTANT && estyle != NONE)
    error->all(FLERR,"Cannot use variable energy with "
               "constant force in fix addforce");
  if ((varflag == EQUAL || varflag == ATOM) &&
      update->whichflag == 2 && estyle == NONE)
    error->all(FLERR,"Must use variable energy with fix addforce");

  if (strstr(update->integrate_style,"respa")) {
    ilevel_respa = ((Respa *) update->integrate)->nlevels-1;
    if (respa_level >= 0) ilevel_respa = MIN(respa_level,ilevel_respa);
  }*/
  /*struct interpol_input {const int in_nx; const int in_ny; const int in_nz; const int in_xmin; const int in_xmax; const int in_ymin; const int in_ymax; const float in_zmin; const int in_zmax; const float in_hx; const float in_hy; const float in_hz;};
  
  vector <interpol_input> I;
  vector < vector < vector<int> > > test_x;
  vector < vector < vector<int> > > test_y;
  vector < vector < vector<int> > > test_z;*/
  
 
  
  
  test_x.clear();
  std::vector < std::vector < std::vector<double> > >().swap(test_x);
  test_y.clear();
  std::vector < std::vector < std::vector<double> > >().swap(test_y);
  test_z.clear();
  std::vector < std::vector < std::vector<double> > >().swap(test_z);
  I.clear();
  vector <interpolar_input>().swap(I);
  data_sigma_Cl.clear();
  std::vector<double>().swap(data_sigma_Cl);
  data_mean_Cl.clear();
  std::vector<double>().swap(data_mean_Cl);
  data_min_sym_Cl.clear();
  std::vector<double>().swap(data_min_sym_Cl);
  data_max_sym_Cl.clear();
  std::vector<double>().swap(data_max_sym_Cl);
  
  data_sigma_Na.clear();
  std::vector<double>().swap(data_sigma_Na);
  data_mean_Na.clear();
  std::vector<double>().swap(data_mean_Na);
  data_min_sym_Na.clear();
  std::vector<double>().swap(data_min_sym_Na);
  data_max_sym_Na.clear();
  std::vector<double>().swap(data_max_sym_Na);
  
  data_dsigma_Cl.clear();
  std::vector<double>().swap(data_dsigma_Cl);
  data_dmean_Cl.clear();
  std::vector<double>().swap(data_dmean_Cl);
  data_dmin_sym_Cl.clear();
  std::vector<double>().swap(data_dmin_sym_Cl);
  data_dmax_sym_Cl.clear();
  std::vector<double>().swap(data_dmax_sym_Cl);
  
  data_dsigma_Na.clear();
  std::vector<double>().swap(data_dsigma_Na);
  data_dmean_Na.clear();
  std::vector<double>().swap(data_dmean_Na);
  data_dmin_sym_Na.clear();
  std::vector<double>().swap(data_dmin_sym_Na);
  data_dmax_sym_Na.clear();
  std::vector<double>().swap(data_dmax_sym_Na);
    
  vector <double> data_x;
  data_x.clear();
  vector<double>().swap(data_x);
  vector <double> data_y;
  data_y.clear();
  vector<double>().swap(data_y);
  vector <double> data_z;
  data_z.clear();
  vector<double>().swap(data_z);
  
    //ifstream file_x("/temp/ema/n2p2_lammps_newsym_1/n2p2/examples/interface-LAMMPS/our_onlyions/amaury_x.dx");//odpre datoteko
    string snx, sxmin, shx;
    string sny, symin,shy;
    string snz, szmin,shz;
    string sefield_x;
    double efield_x;
    string sefield_y;
    double efield_y;
    string sefield_z;
    double efield_z;
    string smean;
    double mean;
    string ssigma;
    double sigma;
    string smin_sym;
    double min_sym;
    string smax_sym;
    double max_sym;
    string sefield_zkonec;
    double efield_zkonec;
    ifstream file_x(edens_x);
    //cout <<edens_x<<"   tz"<<endl;
    string line_x; //naredi prostor, kjer se shranijo podatki in sicer string object
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
             sny = v [6];
             snz = v[7];

        
        }
        if (v[0] == "origin"){
            sxmin = v[1];
            symin = v [2];
            szmin = v[3];
            /*sxmin = 0.0;
            symin = 0.0;
            szmin = 0.0;*/
            
        }
        if (v[0] == "delta"){
            if (v[1] != "0.000000e+00"){
            shx = v[1];}
            if (v[2] != "0.000000e+00"){
            shy = v [2];}
            if (v[3] != "0.000000e+00"){
                shz = v[3];
                }
        }
       if (v[0] != "object" &&  v[0] != "origin" && v[0] != "delta"&& v[0] != "attribute"&& v[0] != "component"){
           for (int index = 0; index < v.size(); ++index) {
               //cout << v[index] << endl;
               sefield_x = v[index];
               stringstream gsefield(sefield_x);
               efield_x = 0.0;
               gsefield >> efield_x;
               data_x.push_back(efield_x);}} //DODALA E_UNIT
    }
    file_x.close();
    // <<"file_closed"<<"size of data_x"<<data_x.size()<<endl;
    //ifstream file_y("/temp/ema/n2p2_lammps_newsym_1/n2p2/examples/interface-LAMMPS/our_onlyions/amaury_y.dx");//odpre datoteko
    
    
   ifstream file_scale_Cl(scale_data_Cl);
    string line_scale_Cl; //naredi prostor, kjer se shranijo podatki in sicer string object
    while (getline(file_scale_Cl, line_scale_Cl)){
        vector <string> v_scale;
        v_scale.clear();
        istringstream ss(line_scale_Cl);
        string word_scale;
        while (ss >> word_scale) {
        v_scale.push_back(word_scale);
        }
            //for (int index = 0; index < v_scale.size(); ++index) {
                
               //cout << v[index] << endl;
               smean = v_scale[0];
               ssigma = v_scale[1];
               smin_sym = v_scale[2];
               smax_sym = v_scale[3];
               stringstream gsmean(smean);
               stringstream gssigma(ssigma);
               stringstream gsmin_sym(smin_sym);
               stringstream gsmax_sym(smax_sym);
               mean = 0.0;
               sigma = 0.0;
               min_sym = 0.0;
               max_sym = 0.0;
               gsmean >> mean;
               gssigma >> sigma;
               gsmin_sym >> min_sym;
               gsmax_sym >> max_sym;
               data_mean_Cl.push_back(mean);
               data_sigma_Cl.push_back(sigma);
               data_min_sym_Cl.push_back(min_sym);
               data_max_sym_Cl.push_back(max_sym);
            //} //DODALA E_UNIT
    }
    cout<<"len sigma Cl"<<data_sigma_Cl.size()<<endl;
    cout<<"len mean Cl"<<data_mean_Cl.size()<<endl;
    file_scale_Cl.close();
    
    ifstream file_dscale_Cl(dscale_data_Cl);
    string line_dscale_Cl; //naredi prostor, kjer se shranijo podatki in sicer string object
    while (getline(file_dscale_Cl, line_dscale_Cl)){
        vector <string> v_scale;
        v_scale.clear();
        istringstream ss(line_dscale_Cl);
        string word_scale;
        while (ss >> word_scale) {
        v_scale.push_back(word_scale);
        }
            //for (int index = 0; index < v_scale.size(); ++index) {
                
               //cout << v[index] << endl;
               smean = v_scale[0];
               ssigma = v_scale[1];
               smin_sym = v_scale[2];
               smax_sym = v_scale[3];
               stringstream gsmean(smean);
               stringstream gssigma(ssigma);
               stringstream gsmin_sym(smin_sym);
               stringstream gsmax_sym(smax_sym);
               mean = 0.0;
               sigma = 0.0;
               min_sym = 0.0;
               max_sym = 0.0;
               gsmean >> mean;
               gssigma >> sigma;
               gsmin_sym >> min_sym;
               gsmax_sym >> max_sym;
               data_dmean_Cl.push_back(mean);
               data_dsigma_Cl.push_back(sigma);
               data_dmin_sym_Cl.push_back(min_sym);
               data_dmax_sym_Cl.push_back(max_sym);
            //} //DODALA E_UNIT
    }
    cout<<"len dsigma Cl"<<data_dsigma_Cl.size()<<endl;
    cout<<"len dmean Cl"<<data_dmean_Cl.size()<<endl;
    file_dscale_Cl.close();
    
    ifstream file_scale_Na(scale_data_Na);
    string line_scale_Na; //naredi prostor, kjer se shranijo podatki in sicer string object
    while (getline(file_scale_Na, line_scale_Na)){
        vector <string> v_scale;
        v_scale.clear();
        istringstream ss(line_scale_Na);
        string word_scale;
        while (ss >> word_scale) {
        v_scale.push_back(word_scale);
        }
            //for (int index = 0; index < v_scale.size(); ++index) {
               //cout << v[index] << endl;
               smean = v_scale[0];
               ssigma = v_scale[1];
               smin_sym = v_scale[2];
               smax_sym = v_scale[3];
               stringstream gsmean(smean);
               stringstream gssigma(ssigma);
               stringstream gsmin_sym(smin_sym);
               stringstream gsmax_sym(smax_sym);
               mean = 0.0;
               sigma = 0.0;
               min_sym = 0.0;
               max_sym = 0.0;
               gsmean >> mean;
               gssigma >> sigma;
               gsmin_sym >> min_sym;
               gsmax_sym >> max_sym;
               data_mean_Na.push_back(mean);
               data_sigma_Na.push_back(sigma);
               data_min_sym_Na.push_back(min_sym);
               data_max_sym_Na.push_back(max_sym);
           // } //DODALA E_UNIT
    }
    cout<<"len sigma Na"<<data_sigma_Na.size()<<endl;
    cout<<"len dmean Na"<<data_mean_Na.size()<<endl;
    file_scale_Na.close();
    
    ifstream file_dscale_Na(dscale_data_Na);
    string line_dscale_Na; //naredi prostor, kjer se shranijo podatki in sicer string object
    while (getline(file_dscale_Na, line_dscale_Na)){
        vector <string> v_scale;
        v_scale.clear();
        istringstream ss(line_dscale_Na);
        string word_scale;
        while (ss >> word_scale) {
        v_scale.push_back(word_scale);
        }
            //for (int index = 0; index < v_scale.size(); ++index) {
               //cout << v[index] << endl;
               smean = v_scale[0];
               ssigma = v_scale[1];
               smin_sym = v_scale[2];
               smax_sym = v_scale[3];
               stringstream gsmean(smean);
               stringstream gssigma(ssigma);
               stringstream gsmin_sym(smin_sym);
               stringstream gsmax_sym(smax_sym);
               mean = 0.0;
               sigma = 0.0;
               min_sym = 0.0;
               max_sym = 0.0;
               gsmean >> mean;
               gssigma >> sigma;
               gsmin_sym >> min_sym;
               gsmax_sym >> max_sym;
               data_dmean_Na.push_back(mean);
               data_dsigma_Na.push_back(sigma);
               data_dmin_sym_Na.push_back(min_sym);
               data_dmax_sym_Na.push_back(max_sym);
           // } //DODALA E_UNIT
    }
    cout<<"len dsigma Na"<<data_dsigma_Na.size()<<endl;
    cout<<"len dmean Na"<<data_dmean_Na.size()<<endl;
    file_dscale_Na.close();
    
    //cout <<"file_closed"<<"size of data_z "<<data_z.size()<<endl;
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
    
    //float hx, hy, hz = 0.0;
    gsnx >> nx;
    gsny >> ny;
    gsnz >> nz;
    gsxmin >> xmin;
    gsymin >> ymin;
    gszmin >> zmin;
    gshx >> hx;
    gshy >> hy;
    gshz >> hz;
    /*xmin = xmin;
    ymin = ymin;
    zmin = zmin;*/
    
    double xmax=xmin+(double(nx-1))*hx;
    double ymax=ymin+(double(ny-1))*hy;
    double zmax=zmin+(double(nz-1))*hz;
    //cout <<"xmin "<<xmin<<"xmax "<<xmax<<"ymin "<<ymin<<"ymax "<<ymax<<"zmin "<<zmin<<"zmax "<<zmax<<endl;
    //cout<<"hx "<<hx<<"hy "<<hy<<"hz "<<hz<<"nx "<<nx<<"ny "<<ny<<"nz "<<nz<<endl;
    /*cout << "type of nx "<<typeid(nx).name() << endl;
    cout << "type of ny "<<typeid(ny).name() << endl;
    cout << "type of nz "<<typeid(nz).name() << endl;*/
    I.push_back({nx, ny, nz, xmin, xmax,ymin, ymax, zmin, zmax, hx, hy, hz});
    cout <<"xmin "<<I[0].in_xmin<<"xmax "<<I[0].in_xmax<<"ymin "<<I[0].in_ymin<<"ymax "<<I[0].in_ymax<<"zmin "<<I[0].in_zmin<<"zmax "<<I[0].in_zmax<<endl;
    cout<<"hx "<<I[0].in_hx<<"hy "<<I[0].in_hy<<"hz "<<I[0].in_hz<<"nx "<<I[0].in_nx<<"ny "<<I[0].in_ny<<"nz "<<I[0].in_nz<<endl;
    //cout <<"here"<<endl;
    int countLines= 0;
    cout <<"here"<<endl;
    
    for (int i = 0; i < nx; ++i) {
        test_x.push_back(vector<vector<double>>());
      
        for (int j = 0; j < ny; ++j) {
            test_x[i].push_back(vector<double>());
           
            
            for (int k = 0; k <nz; ++k) {
                countLines++;
                test_x[i][j].push_back(data_x[countLines-1]);
                
                
            }
        }
    }
    /*cout << test_x[0][0].size()<<" "<< test_x[0].size()<<" "<<test_x.size()<<endl;
    cout << test_y[0][0].size()<<" "<< test_y[0].size()<<" "<<test_y.size()<<endl;
    cout << test_z[0][0].size()<<" "<< test_z[0].size()<<" "<<test_z.size()<<endl;*/
     cout <<"here2"<<endl;
    stringstream gsx_Com(sx_Com);
    stringstream gsy_Com(sy_Com);
    stringstream gsz_Com(sz_Com);
    stringstream gssmin(ssmin);
    stringstream gssmax(ssmax);
    stringstream gsdelta(sdelta);
    stringstream gsE_unit_kcal(sE_unit_kcal);
    stringstream gstype_Cl(stype_Cl);
    stringstream gstype_Na(stype_Na);
    stringstream gstype_Cm(stype_Cm);
    int x_Com_val = 0;
    int y_Com_val=0;
    int z_Com_val=0;
    double smin_val=0.0;
    double smax_val=0.0;
    double delta_val = 0.0;
    double E_unit_kcal_val= 0.0;
    int type_Cl_val=0;
    int type_Na_val=0;
    int type_Cm_val=0;
    //float hx, hy, hz = 0.0;
    gsx_Com >> x_Com_val;
    gsy_Com >> y_Com_val;
    gsz_Com >> z_Com_val;
    gssmin >> smin_val;
    gssmax >> smax_val;
    gsdelta >> delta_val;
    gsE_unit_kcal >> E_unit_kcal_val;
    gstype_Cl >> type_Cl_val;
    gstype_Na >> type_Na_val;
    gstype_Cm >> type_Cm_val;
    x_Com = x_Com_val;
    y_Com = y_Com_val;
    z_Com = z_Com_val;
    smin = smin_val;
    smax = smax_val;
    E_unit_kcal = E_unit_kcal_val;
    delta= delta_val;
    type_Cl = type_Cl_val;
    type_Na = type_Na_val;
    type_Cm = type_Cm_val;
    cout<<x_Com<<" "<<y_Com<<" "<<z_Com<<" "<<delta<<" "<<E_unit_kcal<<" "<<smin<<" "<<smax<<" "<<type_Cl<<" "<<type_Na<<"  "<<type_Cm<<endl;
//test_x.clear();
//test_y.clear();
//test_z.clear();

}
/* ---------------------------------------------------------------------- */

/*void FixAddForceEma::setup(int vflag) //če bi imela kakšen drugačen način update ne verlet; called immediately before the 1st timestep
{
  if (strstr(update->integrate_style,"verlet"))
    pre_force(vflag);
  else {
    ((Respa *) update->integrate)->copy_flevel_f(ilevel_respa);
    post_force_respa(vflag,ilevel_respa,0);
    ((Respa *) update->integrate)->copy_f_flevel(ilevel_respa);
  }
}*/
void FixAddForceEmaCenter::setup_pre_force(int /*vflag*/)
{
  efield_calc();
}
void FixAddForceEmaCenter::pre_force(int /*vflag*/)
{
  if (nevery == 0) return;
  if (update->ntimestep % nevery) return;
  efield_calc();
}


/* ---------------------------------------------------------------------- */

void FixAddForceEmaCenter::min_setup(int vflag)
{
  efield_calc();
}

/* ---------------------------------------------------------------------- */
/*za metodo ki si jo izberes in jo klices*/
void FixAddForceEmaCenter::efield_calc()//pre_force(int ) //(int vflag)  called after pair &molecular forces are computed
{//cout <<"fix_addforce_preforce"<<endl;
//atom je shranjen v Pointers class (pointers.h); vse o sisitemu
  //cout << "in_postforce part"<<endl;
    //if (update->ntimestep % nevery) return;
  //cout <<"befor time"<<endl;
  //if (timestep == 1 PREVERI ALI 0 ALI 1) return;
  //cout <<update->ntimestep<<endl;
  //cout <<"time"<<endl;
  //double delta = 5; //A
  double x_force,y_force,z_force;
  double **x = atom->x;
  double **f = atom->f;
  
  //MOGOCE IZBIRISI?! NULL
  /*double *efield_x = NULL;
  double *efield_y =NULL;
  double *efield_z =NULL;
  
  double *efield1_x = NULL;
  double *efield1_y =NULL;
  double *efield1_z =NULL;
  
  double *efield2_x = NULL;
  double *efield2_y =NULL;
  double *efield2_z =NULL;
  
  double *efield3_x = NULL;
  double *efield3_y =NULL;
  double *efield3_z =NULL;
  
  double *efield4_x = NULL;
  double *efield4_y =NULL;
  double *efield4_z =NULL;
  
  double *efield5_x = NULL;
  double *efield5_y =NULL;
  double *efield5_z =NULL;
  
  double *efield6_x = NULL;
  double *efield6_y =NULL;
  double *efield6_z =NULL;
  
  double *efield7_x = NULL;
  double *efield7_y =NULL;
  double *efield7_z =NULL;
  
  double *efield8_x = NULL;
  double *efield8_y =NULL;
  double *efield8_z =NULL;
 */

  int tmp = 1;
  int ret = atom->find_custom("efieldx",tmp,tmp);
  int rety = atom->find_custom("efieldy",tmp,tmp);
  int retz = atom->find_custom("efieldz",tmp,tmp);
  int ret1 = atom->find_custom("efield1x",tmp,tmp);
  int ret1y = atom->find_custom("efield1y",tmp,tmp);
  int ret1z = atom->find_custom("efield1z",tmp,tmp);
  int ret2 = atom->find_custom("efield2x",tmp,tmp);
  int ret2y = atom->find_custom("efield2y",tmp,tmp);
  int ret2z = atom->find_custom("efield2z",tmp,tmp);
  int ret3 = atom->find_custom("efield3x",tmp,tmp);
  int ret3y = atom->find_custom("efield3y",tmp,tmp);
  int ret3z = atom->find_custom("efield3z",tmp,tmp);
  int ret4 = atom->find_custom("efield4x",tmp,tmp);
  int ret4y = atom->find_custom("efield4y",tmp,tmp);
  int ret4z = atom->find_custom("efield4z",tmp,tmp);
  int ret5 = atom->find_custom("efield5x",tmp,tmp);
  int ret5y = atom->find_custom("efield5y",tmp,tmp);
  int ret5z = atom->find_custom("efield5z",tmp,tmp);
  int ret6 = atom->find_custom("efield6x",tmp,tmp);
  int ret6y = atom->find_custom("efield6y",tmp,tmp);
  int ret6z = atom->find_custom("efield6z",tmp,tmp);
  int ret7 = atom->find_custom("efield7x",tmp,tmp);
  int ret7y = atom->find_custom("efield7y",tmp,tmp);
  int ret7z = atom->find_custom("efield7z",tmp,tmp);
  int ret8 = atom->find_custom("efield8x",tmp,tmp);
  int ret8y = atom->find_custom("efield8y",tmp,tmp);
  int ret8z = atom->find_custom("efield8z",tmp,tmp);
   
  int dret = atom->find_custom("defieldx",tmp,tmp);
  int drety = atom->find_custom("defieldy",tmp,tmp);
  int dretz = atom->find_custom("defieldz",tmp,tmp);
  int dret1 = atom->find_custom("defield1x",tmp,tmp);
  int dret1y = atom->find_custom("defield1y",tmp,tmp);
  int dret1z = atom->find_custom("defield1z",tmp,tmp);
  int dret2 = atom->find_custom("defield2x",tmp,tmp);
  int dret2y = atom->find_custom("defield2y",tmp,tmp);
  int dret2z = atom->find_custom("defield2z",tmp,tmp);
  int dret3 = atom->find_custom("defield3x",tmp,tmp);
  int dret3y = atom->find_custom("defield3y",tmp,tmp);
  int dret3z = atom->find_custom("defield3z",tmp,tmp);
  int dret4 = atom->find_custom("defield4x",tmp,tmp);
  int dret4y = atom->find_custom("defield4y",tmp,tmp);
  int dret4z = atom->find_custom("defield4z",tmp,tmp);
  int dret5 = atom->find_custom("defield5x",tmp,tmp);
  int dret5y = atom->find_custom("defield5y",tmp,tmp);
  int dret5z = atom->find_custom("defield5z",tmp,tmp);
  int dret6 = atom->find_custom("defield6x",tmp,tmp);
  int dret6y = atom->find_custom("defield6y",tmp,tmp);
  int dret6z = atom->find_custom("defield6z",tmp,tmp);
  int dret7 = atom->find_custom("defield7x",tmp,tmp);
  int dret7y = atom->find_custom("defield7y",tmp,tmp);
  int dret7z = atom->find_custom("defield7z",tmp,tmp);
  int dret8 = atom->find_custom("defield8x",tmp,tmp);
  int dret8y = atom->find_custom("defield8y",tmp,tmp);
  int dret8z = atom->find_custom("defield8z",tmp,tmp);

  /*efield_x = atom->dvector[ret];
  efield_y = atom->dvector[rety];
  efield_z = atom->dvector[retz];
  efield1_x = atom->dvector[ret1];
  efield1_y = atom->dvector[ret1y];
  efield1_z = atom->dvector[ret1z];
  efield2_x = atom->dvector[ret2];
  efield2_y = atom->dvector[ret2y];
  efield2_z = atom->dvector[ret2z];
  efield3_x = atom->dvector[ret3];
  efield3_y = atom->dvector[ret3y];
  efield3_z = atom->dvector[ret3z];
  efield4_x = atom->dvector[ret4];
  efield4_y = atom->dvector[ret4y];
  efield4_z = atom->dvector[ret4z];
  efield5_x = atom->dvector[ret5];
  efield5_y = atom->dvector[ret5y];
  efield5_z = atom->dvector[ret5z];
  efield6_x = atom->dvector[ret6];
  efield6_y = atom->dvector[ret6y];
  efield6_z = atom->dvector[ret6z];
  efield7_x = atom->dvector[ret7];
  efield7_y = atom->dvector[ret7y];
  efield7_z = atom->dvector[ret7z];
  efield8_x = atom->dvector[ret8];
  efield8_y = atom->dvector[ret8y];
  efield8_z = atom->dvector[ret8z];*/
  //TUDI TO!
  
  double *efield_x = atom->dvector[ret];
  double *efield_y = atom->dvector[rety];
  double *efield_z = atom->dvector[retz];
  double *efield1_x = atom->dvector[ret1];
  double *efield1_y = atom->dvector[ret1y];
  double *efield1_z = atom->dvector[ret1z];
  double *efield2_x = atom->dvector[ret2];
  double *efield2_y = atom->dvector[ret2y];
  double *efield2_z = atom->dvector[ret2z];
  double *efield3_x = atom->dvector[ret3];
  double *efield3_y = atom->dvector[ret3y];
  double *efield3_z = atom->dvector[ret3z];
  double *efield4_x = atom->dvector[ret4];
  double *efield4_y = atom->dvector[ret4y];
  double *efield4_z = atom->dvector[ret4z];
  double *efield5_x = atom->dvector[ret5];
  double *efield5_y = atom->dvector[ret5y];
  double *efield5_z = atom->dvector[ret5z];
  double *efield6_x = atom->dvector[ret6];
  double *efield6_y = atom->dvector[ret6y];
  double *efield6_z = atom->dvector[ret6z];
  double *efield7_x = atom->dvector[ret7];
  double *efield7_y = atom->dvector[ret7y];
  double *efield7_z = atom->dvector[ret7z];
  double *efield8_x = atom->dvector[ret8];
  double *efield8_y = atom->dvector[ret8y];
  double *efield8_z = atom->dvector[ret8z];

  double *defield_x = atom->dvector[dret];
  double *defield_y = atom->dvector[drety];
  double *defield_z = atom->dvector[dretz];
  double *defield1_x = atom->dvector[dret1];
  double *defield1_y = atom->dvector[dret1y];
  double *defield1_z = atom->dvector[dret1z];
  double *defield2_x = atom->dvector[dret2];
  double *defield2_y = atom->dvector[dret2y];
  double *defield2_z = atom->dvector[dret2z];
  double *defield3_x = atom->dvector[dret3];
  double *defield3_y = atom->dvector[dret3y];
  double *defield3_z = atom->dvector[dret3z];
  double *defield4_x = atom->dvector[dret4];
  double *defield4_y = atom->dvector[dret4y];
  double *defield4_z = atom->dvector[dret4z];
  double *defield5_x = atom->dvector[dret5];
  double *defield5_y = atom->dvector[dret5y];
  double *defield5_z = atom->dvector[dret5z];
  double *defield6_x = atom->dvector[dret6];
  double *defield6_y = atom->dvector[dret6y];
  double *defield6_z = atom->dvector[dret6z];
  double *defield7_x = atom->dvector[dret7];
  double *defield7_y = atom->dvector[dret7y];
  double *defield7_z = atom->dvector[dret7z];
  double *defield8_x = atom->dvector[dret8];
  double *defield8_y = atom->dvector[dret8y];
  double *defield8_z = atom->dvector[dret8z];
 
  
 
  
  
  tagint *id = atom->tag;
  int *mask = atom->mask;
  int *type = atom->type;
  double *q = atom->q;
  imageint *image = atom->image;
  //double v[6];
  int nlocal = atom->nlocal;
 
  cout<<" nlocal fix"<<nlocal<<endl;
  //cout <<ema_var2<<" "<<"ema_var2"<<endl;
  //int b = ema_var2+2;
  //cout <<b<<" "<<"works for ema_var2"<<endl;
  // energy and virial setup

  //if (vflag) v_setup(vflag); DODALA ZA VFLAG
  //else evflag = 0;

  //if (lmp->kokkos)
   // atom->sync_modify(Host, (unsigned int) (F_MASK | MASK_MASK),
       //               (unsigned int) F_MASK);

  // update region if necessary

  /*Region *region = nullptr;
  if (iregion >= 0) {
    region = domain->regions[iregion];
    region->prematch();
  }*/

  // reallocate sforce array if necessary

  //if ((varflag == ATOM || estyle == ATOM) && atom->nmax > maxatom) {
   // maxatom = atom->nmax;
   // memory->destroy(sforce);
   // memory->create(sforce,maxatom,4,"addforce:sforce");
  //}

  // foriginal[0] = "potential energy" for added force
  // foriginal[123] = force on atoms before extra force added
 //POTENCIALNA ENERGIJA JE POSTAVLJENA NA 0 IN SILE DO SEDAJ TUDI!
  //foriginal[0] = foriginal[1] = foriginal[2] = foriginal[3] = 0.0;
  //force_flag = 0;
  
//PAZI INDEXI DELCEV SE SPREMINJAJO! 
 /*double x_Com = 49.86; //A
 double y_Com = 50.04;
 double z_Com = 17.0;*/
 
//double x_Com = 40.0; //A
// double y_Com = 40.0;
// double z_Com = 17.0;
 //int smin = -1;
 //int smax = 1;
 
  
 //double E_unit_kcal = 0.5949576;
 
    double unwrap[3];
    //int q;
    double x_pos,y_pos,z_pos;
    double x_pos1,y_pos1,z_pos1;
    double x_delta, y_delta, z_delta;
    double x_Cm, y_Cm, z_Cm;
    size_t ihi, jhi, khi, ilo, jlo, klo;
    int id_atom;
    for (int index = 0; index < nlocal; index++){
        if (type[index] == type_Cm){
            x_Cm = x[index][0];
            y_Cm = x[index][1];
            z_Cm = x[index][2];
            //cout << "x y z"<<x_Cm<<y_Cm<<z_Cm;
            /*x_delta = x_Cm -x_Com;
            y_delta = y_Cm -y_Com;
            z_delta = z_Cm -z_Com;*/
            x_delta = -(x_Cm -x_Com); //1. premaknemo kot vse na sredisce 0.0.0 2. vektor radzalije moving beada do centra 0.0.0 kjer bo njegova nova pozicija - sredisce mreze  rbeadcenter = (0-pos)
            y_delta = -(y_Cm -y_Com);
            z_delta = -(z_Cm -z_Com);
           // cout << "x y z"<<x_delta<<y_delta<<z_delta;
        }}
    //cout << "before calculation"<<endl;
    for (int index = 0; index < nlocal; index++){ //groupbit ce je atom te group_namev mask je pa shranjen dodaten bit, ki pove kateri skupini pripada atom  class Pointers contains instance of class Atom. Class atom encapsulates atoms positions, velocities, forces, etc. 
      if (mask[index] & groupbit) {
        //if (region && !region->match(x[i][0],x[i][1],x[i][2])) continue;
        domain->unmap(x[index],image[index],unwrap);
        /*if (type[i] == 1){q = 1;} //type 1 je Na type 2 je Cl
        if (type[i] == 2){q = -1;}*/
        //if (type[index] == 3){error->all(FLERR,"bead type for efield");}
        x_pos1 = x[index][0]; //pos v angrstremih
        y_pos1 = x[index][1];
        z_pos1 = x[index][2];
        
        id_atom = id[index];
        
        double charg = q[index];
        //cout<<x_pos<<y_pos<<z_pos<<endl;
        double ifloat, jfloat, kfloat;
        double u_x,u_y,u_z;
        //cout << "before calculation"<<endl;
        /*ifloat = (x_pos-I[0].in_xmin)/I[0].in_hx;
        jfloat = (y_pos-I[0].in_ymin)/I[0].in_hy;
        kfloat = (z_pos-I[0].in_zmin)/I[0].in_hz;*/
        //cout << "after calculation"<<endl;
        /*ifloat = (x_pos-xmin)/hx;
        jfloat = (y_pos-ymin)/hy;
        kfloat = (z_pos-zmin)/hz;*/
        /*x_pos = (x_pos1 -x_Com) -x_delta;
        y_pos = (y_pos1 -y_Com) -y_delta;
        z_pos = (z_pos1 -z_Com) -z_delta;*/
        x_pos = (x_pos1 -x_Com) +x_delta; // 1.vse delce torej Na Cl Cm premkanemo za Com da je sredisce sedaj v 0.0.0 2. vekotor razdalije od beada do iona posion-posbead 
        y_pos = (y_pos1 -y_Com) +y_delta;
        z_pos = (z_pos1 -z_Com) +z_delta;
        double x_posrel = x_pos-0;//-x_Com;
        double y_posrel = y_pos-0;//-y_Com;
        double z_posrel = z_pos-0;//-z_Com;
        
        
        if (x_pos < I[0].in_xmin-x_Com){x_pos = x_pos + (I[0].in_xmax -I[0].in_xmin);}
        if (x_pos > I[0].in_xmax-x_Com){x_pos = x_pos - (I[0].in_xmax -I[0].in_xmin);}
        
        if (y_pos < I[0].in_ymin-y_Com){y_pos = y_pos + (I[0].in_ymax -I[0].in_ymin);}
        if (y_pos > I[0].in_ymax-y_Com){y_pos = y_pos - (I[0].in_ymax -I[0].in_ymin);}
        
        if (z_pos < I[0].in_zmin-z_Com){z_pos = z_pos + (I[0].in_zmax -I[0].in_zmin);}
        if (z_pos > I[0].in_zmax- z_Com){z_pos = z_pos - (I[0].in_zmax -I[0].in_zmin);}
        
        vector <double> x_baze;
        vector <double> z_baze;
        vector <double> y_baze;
        
        //double len_pos = sqrt(x_posrel*x_posrel+y_posrel*y_posrel+z_posrel*z_posrel);
        
            
        //x_baze = {x_posrel/len_pos, y_posrel/len_pos, z_posrel/len_pos};
        //z_baze = {0,0,1};
        //y_baze;
        if ((type[index] == type_Na) || (type[index]== type_Cl)){
            double len_pos = sqrt(x_posrel*x_posrel+y_posrel*y_posrel+z_posrel*z_posrel);
            x_baze = {x_posrel/len_pos, y_posrel/len_pos, z_posrel/len_pos};
            //z_baze = {0,0,1};
            double z1 = 0.1;
            double z2 = 0.1;
            double z3 = -(x_baze[0]*z1+x_baze[1]*z2)/x_baze[2];
            double len_z = sqrt(z1*z1+z2*z2+z3*z3);
            z_baze = {z1/len_z,z2/len_z,z3/len_z};
            
            double i = (z_baze[1]*x_baze[2]-z_baze[2]*x_baze[1]);
            double j = (z_baze[2]*x_baze[0]-z_baze[0]*x_baze[2]);
            double k = (z_baze[0]*x_baze[1]-z_baze[1]*x_baze[0]);
            y_baze = {i,j,k};}
        if (type[index] == type_Cm){
            x_baze = {1,0,0};
            z_baze = {0,0,1};
            
            double i = (z_baze[1]*x_baze[2]-z_baze[2]*x_baze[1]);
            double j = (z_baze[2]*x_baze[0]-z_baze[0]*x_baze[2]);
            double k = (z_baze[0]*x_baze[1]-z_baze[1]*x_baze[0]);
            
            y_baze = {i,j,k};}
            
        
        vector <double> points_x;
        vector <double> points_y;
        vector <double> points_z;
        
        /*if (x_pos < I[0].in_xmin-x_Com){x_pos = x_pos + (I[0].in_xmax -I[0].in_xmin);}
        if (x_pos > I[0].in_xmax-x_Com){x_pos = x_pos - (I[0].in_xmax -I[0].in_xmin);}
        
        if (y_pos < I[0].in_ymin-y_Com){y_pos = y_pos + (I[0].in_ymax -I[0].in_ymin);}
        if (y_pos > I[0].in_ymax-y_Com){y_pos = y_pos - (I[0].in_ymax -I[0].in_ymin);}
        
        if (z_pos < I[0].in_zmin-z_Com){z_pos = z_pos + (I[0].in_zmax -I[0].in_zmin);}
        if (z_pos > I[0].in_zmax- z_Com){z_pos = z_pos - (I[0].in_zmax -I[0].in_zmin);}*/
        
        points_x.push_back(x_pos);
        points_y.push_back(y_pos);
        points_z.push_back(z_pos);
        
        double T1_xpos = x_pos+delta*x_baze[0];
        double T1_ypos = y_pos+delta*x_baze[1];
        double T1_zpos = z_pos+delta*x_baze[2];
        points_x.push_back(T1_xpos);
        points_y.push_back(T1_ypos);
        points_z.push_back(T1_zpos);
        
        double T2_xpos = x_pos+delta*y_baze[0];
        double T2_ypos = y_pos+delta*y_baze[1];
        double T2_zpos = z_pos+delta*y_baze[2];
        points_x.push_back(T2_xpos);
        points_y.push_back(T2_ypos);
        points_z.push_back(T2_zpos);
        
        double T3_xpos = x_pos+delta*z_baze[0];
        double T3_ypos = y_pos+delta*z_baze[1];
        double T3_zpos = z_pos+delta*z_baze[2];
        points_x.push_back(T3_xpos);
        points_y.push_back(T3_ypos);
        points_z.push_back(T3_zpos);
        
        double T4_xpos = x_pos+delta*(-x_baze[0]);
        double T4_ypos = y_pos+delta*(-x_baze[1]);
        double T4_zpos = z_pos+delta*(-x_baze[2]);
        points_x.push_back(T4_xpos);
        points_y.push_back(T4_ypos);
        points_z.push_back(T4_zpos);
        
        double T5_xpos = x_pos+delta*(-y_baze[0]);
        double T5_ypos = y_pos+delta*(-y_baze[1]);
        double T5_zpos = z_pos+delta*(-y_baze[2]);
        points_x.push_back(T5_xpos);
        points_y.push_back(T5_ypos);
        points_z.push_back(T5_zpos);
        
        double T6_xpos = x_pos+delta*(-z_baze[0]);
        double T6_ypos = y_pos+delta*(-z_baze[1]);
        double T6_zpos = z_pos+delta*(-z_baze[2]);
        points_x.push_back(T6_xpos);
        points_y.push_back(T6_ypos);
        points_z.push_back(T6_zpos);
        
        double T7_xpos = x_pos+(delta*x_baze[0]+delta*y_baze[0]);
        double T7_ypos = y_pos+(delta*x_baze[1]+delta*y_baze[1]);
        double T7_zpos = z_pos+(delta*x_baze[2]+delta*y_baze[2]);
        points_x.push_back(T7_xpos);
        points_y.push_back(T7_ypos);
        points_z.push_back(T7_zpos);
        
        double T8_xpos = x_pos-(delta*x_baze[0]+delta*y_baze[0]);
        double T8_ypos = y_pos-(delta*x_baze[1]+delta*y_baze[1]);
        double T8_zpos = z_pos-(delta*x_baze[2]+delta*y_baze[2]);
        points_x.push_back(T8_xpos);
        points_y.push_back(T8_ypos);
        points_z.push_back(T8_zpos);
        
        
        if ((x_pos > (I[0].in_xmax-x_Com  ))|| (x_pos < (I[0].in_xmin-x_Com))) {cout << "x out of range "<<x_pos<< "  "<< I[0].in_xmax-x_Com<<endl; exit(EXIT_FAILURE);}
        if ((y_pos > (I[0].in_ymax- y_Com ) )|| (y_pos < (I[0].in_ymin -y_Com))) {cout << "y out of range "<< y_pos<<endl; exit(EXIT_FAILURE);}
        if ((z_pos > (I[0].in_zmax -z_Com))|| (z_pos < (I[0].in_zmin-z_Com))) {cout << "z out of range "<< z_pos<<" min "<<(I[0].in_zmin-z_Com)<<" max "<<(I[0].in_zmax -z_Com)<<endl; exit(EXIT_FAILURE);}
        
        //cout <<"SIZE"<<points_x.size()<<endl;
        vector <double> list;
        for (int j=0; j<points_x.size();j++){
        size_t ihi, jhi, khi, ilo, jlo, klo;    
       // cout <<" OK3"<<endl;
        double ifloat, jfloat, kfloat;
        double u_x = 0.0;
        double u_y = 0.0;
        double u_z = 0.0;
        double du_x = 0.0;
        double du_y = 0.0;
        double du_z = 0.0;
        //if ((points_x[j] > (I[0].in_xmax-x_Com  ))|| (points_x[j] < (I[0].in_xmin-x_Com))) {cout << "x out of range points "<<x_pos<< "  "<< I[0].in_xmax-x_Com<<endl; exit(EXIT_FAILURE);}
        //if ((points_y[j] > (I[0].in_ymax- y_Com ) )|| (points_y[j] < (I[0].in_ymin -y_Com))) {cout << "y out of range points "<< y_pos<<" "<<(I[0].in_ymin -y_Com)<<" "<<(I[0].in_ymax- y_Com )<<endl; exit(EXIT_FAILURE);}
        //if ((points_z[j] > (I[0].in_zmax -z_Com))|| (points_z[j] < (I[0].in_zmin-z_Com))) {cout << "z out of range points"<< z_pos<<" min "<<(I[0].in_zmin-z_Com)<<" max "<<(I[0].in_zmax -z_Com)<<"  "<<type[index]<<endl;}// exit(EXIT_FAILURE);}
        /*ifloat = (points_x[j]-I[0].in_xmin)/I[0].in_hx;
        jfloat = (points_y[j]-I[0].in_ymin)/I[0].in_hy;
        kfloat = (points_z[j]-I[0].in_zmin)/I[0].in_hz;*/
        
        ifloat = (points_x[j]-(I[0].in_xmin-x_Com))/I[0].in_hx;
        jfloat = (points_y[j]-(I[0].in_ymin-y_Com))/I[0].in_hy;
        kfloat = (points_z[j]-(I[0].in_zmin-z_Com))/I[0].in_hz;
       
        cout <<std::fixed << std::setprecision(8);// << ifloat; //tudi APBS 6
        cout <<std::fixed << std::setprecision(8);// << jfloat;
        cout <<std::fixed << std::setprecision(8);// << kfloat;
        //cout <<"ifloat"<<ifloat<<"jfloat"<<jfloat<<"kfloat"<<kfloat<<endl;

        //if ((points_x[j] > I[0].in_xmin) && (points_y[j] > I[0].in_ymin ) && (points_z[j] > I[0].in_zmin)){
        if ((points_x[j] > (I[0].in_xmin-x_Com)) && (points_y[j] > (I[0].in_ymin-y_Com) ) && (points_z[j] > (I[0].in_zmin-z_Com))){
        ihi = (int)ceil(ifloat);
        ilo = (int)floor(ifloat);
        jhi = (int)ceil(jfloat);
        jlo = (int)floor(jfloat);
        khi = (int)ceil(kfloat);
        klo = (int)floor(kfloat);
        
        int Vgrid_digits = 9; //tudi v APBS toliko 6
        double Vcompare = pow(10,-1*(Vgrid_digits - 2));
        /*if (abs(points_x[j]y - I[0].in_xmin) < Vcompare) {ilo = 0;} //če je razlika med vrednostjo in minimalno vrednostjo manjša od natančnosti števila potem moraš vpisati za max obrat0
        if (abs(points_y[j] - I[0].in_ymin) < Vcompare) {jlo = 0;}
        if (abs(points_z[j] - I[0].in_zmin) < Vcompare) {klo = 0;}
        if (abs(points_x[j] - I[0].in_xmax) < Vcompare) {ihi = I[0].in_nx - 1;}
        if (abs(points_y[j] - I[0].in_ymax) < Vcompare) {jhi = I[0].in_ny - 1;}
        if (abs(points_z[j] - I[0].in_zmax) < Vcompare) {khi = I[0].in_nz - 1;}*/
        
        /*if (abs(points_x[j] - (I[0].in_xmin-x_Com)) < Vcompare) {ilo = 0;} //če je razlika med vrednostjo in minimalno vrednostjo manjša od natančnosti števila potem moraš vpisati za max obrat0
        if (abs(points_y[j] - (I[0].in_ymin-y_Com)) < Vcompare) {jlo = 0;}
        if (abs(points_z[j] - (I[0].in_zmin-z_Com)) < Vcompare) {klo = 0;}
        if (abs(points_x[j] - (I[0].in_xmax-x_Com)) < Vcompare) {ihi = I[0].in_nx - 1;}
        if (abs(points_y[j] - (I[0].in_ymax-y_Com)) < Vcompare) {jhi = I[0].in_ny - 1;}
        if (abs(points_z[j] - (I[0].in_zmax-z_Com)) < Vcompare) {khi = I[0].in_nz - 1;}*/
        
        /*if (points_x[j] < I[0].in_xmin){ ilo = 0, ihi = 0;}
        if (points_x[j] > I[0].in_xmax ){ihi = I[0].in_nx - 1, ilo =I[0].in_nx - 1; }

        if (points_y[j] < I[0].in_ymin ){ jlo = 0, jhi = 0;}
        if (points_y[j] > I[0].in_ymax){ jhi = I[0].in_ny - 1, jlo = I[0].in_ny - 1;}

        if (points_z[j] < I[0].in_zmin){ klo = 0, khi = 0;}
        if (points_z[j] > I[0].in_zmax){khi = I[0].in_nz - 1,klo = I[0].in_nz - 1;}*/
        
        
        
        /*if ((x_pos > (I[0].in_xmax-x_Com)+x_delta )|| (x_pos < (I[0].in_xmin-x_Com)+x_delta)) {cout << "x out of range "<<x_pos<< endl;} //exit(EXIT_FAILURE);}
        if ((y_pos > (I[0].in_ymax-y_Com)+y_delta )|| (y_pos < (I[0].in_ymin-y_Com)+y_delta)) {cout << "y out of range "<< y_pos<<endl;} //exit(EXIT_FAILURE);}
        if ((z_pos > (I[0].in_zmax-z_Com)+z_delta )|| (z_pos < (I[0].in_zmin-z_Com)+z_delta)) {cout << "z out of range "<< z_pos<<endl;} //exit(EXIT_FAILURE);}*/
        if ((ihi<I[0].in_nx) && (jhi<I[0].in_ny) && (khi<I[0].in_nz))// && (ilo>=0.0) && (jlo>=0.0) && (klo>=0.0)) //&& ((ihi != ilo) && (jhi != jlo) && (khi != klo))){
        {
        double dx = (ifloat - double(ilo));
        double dy = (jfloat - double(jlo));
        double dz = (kfloat - double(klo));

        double dx_h = abs(ifloat - double(ihi));
        double dy_h = abs(jfloat - double(jhi));
        double dz_h = abs(kfloat - double(khi));
        
        double r_hx, r_hy, r_hz;
        /*int d ,j, k;
        r_hx = dx_h;d = ihi;
        r_hy = dy_h;j = jhi;
        r_hz = dz_h;k = khi;
        double r_dev = sqrt(r_hx*r_hx+r_hy*r_hy+r_hz*r_hz);*/
         
        
        if ((ihi == ilo) && (jhi == jlo) && (khi == klo)){
            double r_h = sqrt(I[0].in_hx*I[0].in_hx+I[0].in_hy*I[0].in_hy+I[0].in_hz*I[0].in_hz);
            du_x = (test_x[ilo+I[0].in_hx][jlo+I[0].in_hy][klo+I[0].in_hz]-test_x[ilo-I[0].in_hx][jlo-I[0].in_hy][klo-I[0].in_hz])/2*r_h;
            
             u_x = test_x[ilo][jlo][klo];
            
            
        }
        
        
        else {//cout<<"IN THE LOOP FOR NOT ON MESH"<<endl;
        int a1,a2,a3,a4,a5,a6,a7,a8;
        int b1,b2,b3,b4,b5,b6,b7,b8;
        int c1,c2,c3,c4,c5,c6,c7,c8;
        /*if (dx<Vcompare){a1 = a2=a3=a4=a5=a6=a7=a8 = ilo;
        //cout<<"IN THE LOOP FOR NOT ON MESH less then Vcompare MIN "<<endl;
        dx = 0;

        }
        if ((abs(dx_h)<Vcompare)){a1 = a2=a3=a4=a5=a6=a7=a8 = ihi;
        //cout<<"IN THE LOOP FOR NOT ON MESH less then Vcompare max "<<endl;
        //dx = abs(dx_h);
        dx = 1;


        }
        if ((dx>Vcompare) && (abs(dx_h)>Vcompare)){a1 = a2 = a3 = a4 = ihi;
        a5 = a6=a7=a8 = ilo;

        //cout<<"IN THE LOOP FOR NOT ON MESH X "<<endl;

        }
        if (dy<Vcompare){b1 = b2=b3=b4=b5=b6=b7=b8 = jlo;
        dy = 0;
        //cout<<"IN THE LOOP FOR NOT ON MESH less then Vcompare MIN "<<endl;

        }
        if ((abs(dy_h)<Vcompare)){b1 = b2=b3=b4=b5=b6=b7=b8 = jhi;
        //cout<<"IN THE LOOP FOR NOT ON MESH less then Vcompare max "<<endl;
        //dy = abs(dy_h);
        dy = 1;

        }
        if ((dy>Vcompare) && (abs(dy_h)>Vcompare)){b1 = b3 = b5 = b7 = jhi;
        b2 = b4=b6=b8 = jlo;
        //cout<<"IN THE LOOP FOR NOT ON MESH y "<<endl;
        }
        if (dz<Vcompare){c1=c2=c3=c4=c5=c6=c7=c8 = klo;
        //cout<<"IN THE LOOP FOR NOT ON MESH less then Vcompare MIN "<<endl;
        dz = 0;
        }
        if ((abs(dz_h)<Vcompare)){c1=c2=c3=c4=c5=c6=c7=c8 = khi;
        //cout<<"IN THE LOOP FOR NOT ON MESH less then Vcompare max "<<endl;
        //dz = abs(dz_h);
        dz = 1;
        }
        if ((dz>Vcompare) && (abs(dz_h)>Vcompare)){c1=c2=c5=c6= khi;
        c3=c4=c7=c8 = klo;
        //cout<<"IN THE LOOP FOR NOT ON MESH X "<<endl;
        }*/
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
        
        //du_x = (-test_x[d][j][k]*E_unit_kcal-(-u_x*E_unit_kcal))/r_dev;
        //cout << "test_x_ihi_jhi_khi"<<test_x[ihi][jhi][khi];
        //cout <<" OK6_1/2"<<endl;
       
        //cout << "test_z_ihi_jhi_khi"<<test_z[ihi][jhi][khi];
           
    }}
        else{
        //if ((ihi>I[0].in_nx) && (jhi>I[0].in_ny) && (khi>I[0].in_nz))
         // cout<<" in else not less then nx or ny or nz"<<I[0].in_nx<<" "<<I[0].in_ny<<" "<<I[0].in_nz<<" "<<type[index]<<endl;
         //   cout << ifloat <<" "<<jfloat<<" "<<kfloat<<" "<<endl;
         //    cout << x_posrel <<" "<<x_pos<<" "<<y_posrel<<" "<<y_pos<<" "<<z_posrel<<" "<<z_pos<<endl;
        //exit(EXIT_FAILURE);
        u_x = 0.0;
       }
        }
        //if ((points_x[j] < (I[0].in_xmin-x_Com)) && (points_y[j] < (I[0].in_ymin-y_Com) ) && (points_z[j] < (I[0].in_zmin-z_Com)))
        else{
        //cout<<" in else not more then xmin or ymin or zmin"<<I[0].in_xmin<<" "<<I[0].in_ymin<<" "<<I[0].in_zmin<<" "<<x_posrel <<" "<<x_pos<<" "<<points_x[j]<<" "<<y_posrel<<" "<<y_pos<<" "<<points_y[j]<<" "<<z_posrel<<" "<<z_pos<<" "<<points_z[j]<<" "<<type[index]<<endl;
        //exit(EXIT_FAILURE);
        u_x = 0.0;
        }
       
        double x_efield = u_x; //-u_x *E_unit_kcal; //*e; //F[eV/A] =- E[V/A]*q - ker iz APBS je samo gradient pot
        
       //cout <<" OK6_1/3"<<endl;
        //cout<<"fix_addforce_ema_calculati of efiled"<<endl;
        
         
            //cout<<type[index]<<" "<<type_Na<<endl;
        //cout<<type[index]<<" "<<type_Na<<" "<<q[index]<<endl;
        if (type[index] == type_Na){
        if (j == 0){//cout<<"first is ok IN  "<<j<<" xefield "<<x_efield<<" "<<index<<endl;
        efield_x[index]= 700 ;//(x_efield-data_mean_Na[0])/(data_sigma_Na[0]);
        
        efield_y[index]=0.0;
        
        efield_z[index] =0.0;
        
        /* defield_x[index]= smin+((smax-smin)*(du_x-data_dmean_Na[0])/(data_dmax_sym_Na[0]-data_dmin_sym_Na[0]));
      
          defield_y[index]= smin+((smax-smin)*(du_y-data_dmean_Na[1])/(data_dmax_sym_Na[1]-data_dmin_sym_Na[1]));
        //cout<<"first of first is ok IN"<<endl;
        defield_z[index] =smin+((smax-smin)*(du_z-data_dmean_Na[2])/(data_dmax_sym_Na[2]-data_dmin_sym_Na[2]));*/
        //cout<<"first is ok IN_end"<<endl;
        }
        
        if (j == 1){//cout<<"second is ok IN  "<<j<<" xefield "<<x_efield<<" "<<index<<endl;
        efield1_x[index]=100 ;//(x_efield-data_mean_Na[1])/(data_sigma_Na[1]);
        efield1_y[index]= 0.0;
       efield1_z[index] = 0.0;
        /*defield1_x[index]= smin+((smax-smin)*(du_x-data_dmean_Na[3])/(data_dmax_sym_Na[3]-data_dmin_sym_Na[3]));
        defield1_y[index]= smin+((smax-smin)*(du_y-data_dmean_Na[4])/(data_dmax_sym_Na[4]-data_dmin_sym_Na[4]));
        defield1_z[index]= smin+((smax-smin)*(du_z-data_dmean_Na[5])/(data_dmax_sym_Na[5]-data_dmin_sym_Na[5]));*/
        }
        
        if (j == 2){//cout<<"third is ok IN  "<<j<<" xefield "<<x_efield<<" "<<index<<endl;
        efield2_x[index]=100 ;//(x_efield-data_mean_Na[2])/(data_sigma_Na[2]);
       // cout<<"third is ok IN 1"<<endl;
        efield2_y[index]= 0.0;
        //cout<<"third is ok IN 2"<<endl;
        efield2_z[index] = 0.0;
        /*defield2_x[index]= smin+((smax-smin)*(du_x-data_dmean_Na[6])/(data_dmax_sym_Na[6]-data_dmin_sym_Na[6]));
        defield2_y[index]= smin+((smax-smin)*(du_y-data_dmean_Na[7])/(data_dmax_sym_Na[7]-data_dmin_sym_Na[7]));
        defield2_z[index]=smin+((smax-smin)*(du_z-data_dmean_Na[8])/(data_dmax_sym_Na[8]-data_dmin_sym_Na[8]));*/
        //cout<<"third is ok IN 3"<<endl;
        }
         if (j == 3){//cout<<"4 is ok IN  "<<j<<" xefield "<<x_efield<<" "<<index<<endl;
         efield3_x[index]= 100 ;//(x_efield-data_mean_Na[3])/(data_sigma_Na[3]);
        // cout<<"third is ok IN 1"<<endl;
        efield3_y[index]=0.0;
        //cout<<"third is ok IN 2"<<endl;
        efield3_z[index] = 0.0;
        /*defield3_x[index]= smin+((smax-smin)*(du_x-data_dmean_Na[9])/(data_dmax_sym_Na[9]-data_dmin_sym_Na[9]));
        defield3_y[index]= smin+((smax-smin)*(du_y-data_dmean_Na[10])/(data_dmax_sym_Na[10]-data_dmin_sym_Na[10]));
        defield3_z[index]= smin+((smax-smin)*(du_z-data_dmean_Na[11])/(data_dmax_sym_Na[11]-data_dmin_sym_Na[11]));*/
        //cout<<"third is ok IN 3"<<endl;
        }
         if (j == 4){//cout<<" 5 is ok IN  "<<j<<" xefield "<<x_efield<<" "<<index<<endl;
        efield4_x[index]=100 ;//(x_efield-data_mean_Na[4])/(data_sigma_Na[4]);
        //cout<<"third is ok IN 1"<<endl;
        efield4_y[index]=0.0;
        //cout<<"third is ok IN 3"<<endl;
        efield4_z[index] =0.0;
        /*defield4_x[index]= smin+((smax-smin)*(du_x-data_dmean_Na[12])/(data_dmax_sym_Na[12]-data_dmin_sym_Na[12]));
        defield4_y[index]= smin+((smax-smin)*(du_y-data_dmean_Na[13])/(data_dmax_sym_Na[13]-data_dmin_sym_Na[13]));
        defield4_z[index]= smin+((smax-smin)*(du_z-data_dmean_Na[14])/(data_dmax_sym_Na[14]-data_dmin_sym_Na[14]));*/
        }
        //cout<<"third is ok IN 3"<<endl;
        
         if (j == 5){//cout<<"6 is ok IN  "<<j<<" xefield "<<x_efield<<" "<<index<<endl;
        efield5_x[index]=100 ;//(x_efield-data_mean_Na[5])/(data_sigma_Na[5]);
        efield5_y[index]=0.0;
       efield5_z[index] = 0.0;
        /*defield5_x[index]= smin+((smax-smin)*(du_x-data_dmean_Na[15])/(data_dmax_sym_Na[15]-data_dmin_sym_Na[15]));
        defield5_y[index]=smin+((smax-smin)*(du_y-data_dmean_Na[16])/(data_dmax_sym_Na[16]-data_dmin_sym_Na[16]));
        defield5_z[index]= smin+((smax-smin)*(du_z-data_dmean_Na[17])/(data_dmax_sym_Na[17]-data_dmin_sym_Na[17]));*/
        }
        //cout<<"first is ok"<<endl;}
         if (j == 6){//cout<<"7 is ok IN  "<<j<<" xefield "<<x_efield<<" "<<index<<endl;
        efield6_x[index]=100 ;//(x_efield-data_mean_Na[6])/(data_sigma_Na[6]);
        efield6_y[index]=0.0;
        efield6_z[index] =0.0;
        /*defield6_x[index]= smin+((smax-smin)*(du_x-data_dmean_Na[18])/(data_dmax_sym_Na[18]-data_dmin_sym_Na[18]));
        defield6_y[index]= smin+((smax-smin)*(du_y-data_dmean_Na[19])/(data_dmax_sym_Na[19]-data_dmin_sym_Na[19]));
        defield6_z[index]= smin+((smax-smin)*(du_z-data_dmean_Na[20])/(data_dmax_sym_Na[20]-data_dmin_sym_Na[20]));*/
        }
        //cout<<"first is ok"<<endl;}
         if (j == 7){//cout<<"8 is ok IN  "<<j<<" xefield "<<x_efield<<" "<<index<<endl;
        efield7_x[index]=100 ;// (x_efield-data_mean_Na[7])/(data_sigma_Na[7]);
        efield7_y[index]= 0.0;
        efield7_z[index] = 0.0;
        /*defield7_x[index]= smin+((smax-smin)*(du_x-data_dmean_Na[21])/(data_dmax_sym_Na[21]-data_dmin_sym_Na[21]));
        defield7_y[index]= smin+((smax-smin)*(du_y-data_dmean_Na[22])/(data_dmax_sym_Na[22]-data_dmin_sym_Na[22]));
        defield7_z[index]= smin+((smax-smin)*(du_z-data_dmean_Na[23])/(data_dmax_sym_Na[23]-data_dmin_sym_Na[23]));*/
        }
        //cout<<"first is ok"<<endl;}
        if (j == 8){//cout<<"9 is ok IN  "<<j<<" xefield "<<x_efield<<" "<<index<<endl;
        efield8_x[index]=100 ;//(x_efield-data_mean_Na[8])/(data_sigma_Na[8]);
        efield8_y[index]= 0.0;
        efield8_z[index] = 0.0;
        /*defield8_x[index]= smin+((smax-smin)*(du_x-data_dmean_Na[24])/(data_dmax_sym_Na[24]-data_dmin_sym_Na[24]));
        defield8_y[index]=smin+((smax-smin)*(du_y-data_dmean_Na[25])/(data_dmax_sym_Na[25]-data_dmin_sym_Na[25]));
        defield8_z[index]= smin+((smax-smin)*(du_z-data_dmean_Na[26])/(data_dmax_sym_Na[26]-data_dmin_sym_Na[26]));*/
        }}
        
        if (type[index] == type_Cl){
        if (j == 0){//cout<<"first is ok IN  "<<j<<" xefield "<<x_efield<<" "<<index<<endl;
        efield_x[index]=900 ;//(x_efield-data_mean_Cl[0])/(data_sigma_Cl[0]);
        
        efield_y[index]=0.0;
        
        efield_z[index] =0.0;
        
        /* defield_x[index]= smin+((smax-smin)*(du_x-data_dmean_Na[0])/(data_dmax_sym_Na[0]-data_dmin_sym_Na[0]));
      
          defield_y[index]= smin+((smax-smin)*(du_y-data_dmean_Na[1])/(data_dmax_sym_Na[1]-data_dmin_sym_Na[1]));
        //cout<<"first of first is ok IN"<<endl;
        defield_z[index] =smin+((smax-smin)*(du_z-data_dmean_Na[2])/(data_dmax_sym_Na[2]-data_dmin_sym_Na[2]));*/
        //cout<<"first is ok IN_end"<<endl;
        }
        
        if (j == 1){//cout<<"second is ok IN  "<<j<<" xefield "<<x_efield<<" "<<index<<endl;
        efield1_x[index]=100 ;//(x_efield-data_mean_Cl[1])/(data_sigma_Cl[1]);
        efield1_y[index]= 0.0;
       efield1_z[index] = 0.0;
        /*defield1_x[index]= smin+((smax-smin)*(du_x-data_dmean_Na[3])/(data_dmax_sym_Na[3]-data_dmin_sym_Na[3]));
        defield1_y[index]= smin+((smax-smin)*(du_y-data_dmean_Na[4])/(data_dmax_sym_Na[4]-data_dmin_sym_Na[4]));
        defield1_z[index]= smin+((smax-smin)*(du_z-data_dmean_Na[5])/(data_dmax_sym_Na[5]-data_dmin_sym_Na[5]));*/
        }
        
        if (j == 2){//cout<<"third is ok IN  "<<j<<" xefield "<<x_efield<<" "<<index<<endl;
        efield2_x[index]=100 ;//(x_efield-data_mean_Cl[2])/(data_sigma_Cl[2]);
       // cout<<"third is ok IN 1"<<endl;
        efield2_y[index]= 0.0;
        //cout<<"third is ok IN 2"<<endl;
        efield2_z[index] = 0.0;
        /*defield2_x[index]= smin+((smax-smin)*(du_x-data_dmean_Na[6])/(data_dmax_sym_Na[6]-data_dmin_sym_Na[6]));
        defield2_y[index]= smin+((smax-smin)*(du_y-data_dmean_Na[7])/(data_dmax_sym_Na[7]-data_dmin_sym_Na[7]));
        defield2_z[index]=smin+((smax-smin)*(du_z-data_dmean_Na[8])/(data_dmax_sym_Na[8]-data_dmin_sym_Na[8]));*/
        //cout<<"third is ok IN 3"<<endl;
        }
         if (j == 3){//cout<<"4 is ok IN  "<<j<<" xefield "<<x_efield<<" "<<index<<endl;
         efield3_x[index]= 100 ;//(x_efield-data_mean_Cl[3])/(data_sigma_Cl[3]);
        // cout<<"third is ok IN 1"<<endl;
        efield3_y[index]=0.0;
        //cout<<"third is ok IN 2"<<endl;
        efield3_z[index] = 0.0;
        /*defield3_x[index]= smin+((smax-smin)*(du_x-data_dmean_Na[9])/(data_dmax_sym_Na[9]-data_dmin_sym_Na[9]));
        defield3_y[index]= smin+((smax-smin)*(du_y-data_dmean_Na[10])/(data_dmax_sym_Na[10]-data_dmin_sym_Na[10]));
        defield3_z[index]= smin+((smax-smin)*(du_z-data_dmean_Na[11])/(data_dmax_sym_Na[11]-data_dmin_sym_Na[11]));*/
        //cout<<"third is ok IN 3"<<endl;
        }
         if (j == 4){//cout<<" 5 is ok IN  "<<j<<" xefield "<<x_efield<<" "<<index<<endl;
        efield4_x[index]=100 ;//(x_efield-data_mean_Cl[4])/(data_sigma_Cl[4]);
        //cout<<"third is ok IN 1"<<endl;
        efield4_y[index]=0.0;
        //cout<<"third is ok IN 3"<<endl;
        efield4_z[index] =0.0;
        /*defield4_x[index]= smin+((smax-smin)*(du_x-data_dmean_Na[12])/(data_dmax_sym_Na[12]-data_dmin_sym_Na[12]));
        defield4_y[index]= smin+((smax-smin)*(du_y-data_dmean_Na[13])/(data_dmax_sym_Na[13]-data_dmin_sym_Na[13]));
        defield4_z[index]= smin+((smax-smin)*(du_z-data_dmean_Na[14])/(data_dmax_sym_Na[14]-data_dmin_sym_Na[14]));*/
        }
        //cout<<"third is ok IN 3"<<endl;
        
         if (j == 5){//cout<<"6 is ok IN  "<<j<<" xefield "<<x_efield<<" "<<index<<endl;
        efield5_x[index]=100 ;//(x_efield-data_mean_Cl[5])/(data_sigma_Cl[5]);
        efield5_y[index]=0.0;
       efield5_z[index] = 0.0;
        /*defield5_x[index]= smin+((smax-smin)*(du_x-data_dmean_Na[15])/(data_dmax_sym_Na[15]-data_dmin_sym_Na[15]));
        defield5_y[index]=smin+((smax-smin)*(du_y-data_dmean_Na[16])/(data_dmax_sym_Na[16]-data_dmin_sym_Na[16]));
        defield5_z[index]= smin+((smax-smin)*(du_z-data_dmean_Na[17])/(data_dmax_sym_Na[17]-data_dmin_sym_Na[17]));*/
        }
        //cout<<"first is ok"<<endl;}
         if (j == 6){//cout<<"7 is ok IN  "<<j<<" xefield "<<x_efield<<" "<<index<<endl;
        efield6_x[index]=100 ;//(x_efield-data_mean_Cl[6])/(data_sigma_Cl[6]);
        efield6_y[index]=0.0;
        efield6_z[index] =0.0;
        /*defield6_x[index]= smin+((smax-smin)*(du_x-data_dmean_Na[18])/(data_dmax_sym_Na[18]-data_dmin_sym_Na[18]));
        defield6_y[index]= smin+((smax-smin)*(du_y-data_dmean_Na[19])/(data_dmax_sym_Na[19]-data_dmin_sym_Na[19]));
        defield6_z[index]= smin+((smax-smin)*(du_z-data_dmean_Na[20])/(data_dmax_sym_Na[20]-data_dmin_sym_Na[20]));*/
        }
        //cout<<"first is ok"<<endl;}
         if (j == 7){//cout<<"8 is ok IN  "<<j<<" xefield "<<x_efield<<" "<<index<<endl;
        efield7_x[index]= 100 ;//(x_efield-data_mean_Cl[7])/(data_sigma_Cl[7]);
        efield7_y[index]= 0.0;
        efield7_z[index] = 0.0;
        /*defield7_x[index]= smin+((smax-smin)*(du_x-data_dmean_Na[21])/(data_dmax_sym_Na[21]-data_dmin_sym_Na[21]));
        defield7_y[index]= smin+((smax-smin)*(du_y-data_dmean_Na[22])/(data_dmax_sym_Na[22]-data_dmin_sym_Na[22]));
        defield7_z[index]= smin+((smax-smin)*(du_z-data_dmean_Na[23])/(data_dmax_sym_Na[23]-data_dmin_sym_Na[23]));*/
        }
        //cout<<"first is ok"<<endl;}
        if (j == 8){//cout<<"9 is ok IN  "<<j<<" xefield "<<x_efield<<" "<<index<<endl;
        efield8_x[index]=100 ;//(x_efield-data_mean_Cl[8])/(data_sigma_Cl[8]);
        efield8_y[index]= 0.0;
        efield8_z[index] = 0.0;
        /*defield8_x[index]= smin+((smax-smin)*(du_x-data_dmean_Na[24])/(data_dmax_sym_Na[24]-data_dmin_sym_Na[24]));
        defield8_y[index]=smin+((smax-smin)*(du_y-data_dmean_Na[25])/(data_dmax_sym_Na[25]-data_dmin_sym_Na[25]));
        defield8_z[index]= smin+((smax-smin)*(du_z-data_dmean_Na[26])/(data_dmax_sym_Na[26]-data_dmin_sym_Na[26]));*/
        }}
        if (id[index] == 33){efield_x[index]=200 ;//(x_efield-data_mean_Cl[0])/(data_sigma_Cl[0]);
        
        efield_y[id[index]]=300;
        
        efield_z[id[index]] =400;
        efield8_z[id[index]]=500 ;
        }
        if (id[index] == 100){efield_x[id[index]]=200 ;//(x_efield-data_mean_Cl[0])/(data_sigma_Cl[0]);
        
        efield_y[id[index]]=300;
        
        efield_z[id[index]] =400;
        efield8_z[id[index]]=500 ;
        }
        
        }   
        
           
        
}}}
        //cout<<"first is ok"<<endl;}}
       
       /*cout<<"fix_add_force_po_e_x"<<endl;
         if (j <= 2){cout<<"fix_add_force_pres_e_x"<<endl;
             efield_x[index]= x_efield;
        efield_y[index]= y_efield;
        efield_z[index] = z_efield;*/
       /*double e_x = efield_x[index];
        double e_y = efield_y[index];
        double e_z = efield_z[index];
       
        cout<<e_x<<e_y<<e_z<<endl;*/
       /*cout<<"first is ok"<<endl;
        }
        
        if (j <= 5){
         efield1_x[index]= 1.0;
        efield1_y[index]= 2.0;
        efield1_z[index] = 3.0;
        cout<<"second is ok"<<endl;
        }
        if (j <= 8){
         efield2_x[index]= 1;
        efield2_y[index]= 2;
        efield2_z[index] = 3;
        }
         if (j <= 11){
         efield3_x[index]= 1;
        efield3_y[index]= 2;
        efield3_z[index] = 3;
        }
         if (j <= 14){
        efield4_x[index]= 1;
        efield4_y[index]= 2;
        efield4_z[index] = 3;
        }
         if (j <= 17){
        efield5_x[index]= 1;
        efield5_y[index]= 2;
        efield5_z[index] =3;
        }
         if (j <= 20){
        efield6_x[index]=1;
        efield6_y[index]= 2;
        efield6_z[index] = 3;
        }
         if (j <= 23){
        efield7_x[index]= 1;
        efield7_y[index]= 2;
        efield7_z[index] = 3;
        }
         if (j <= 26){
       efield8_x[index]= 1;
        efield8_y[index]= 2;
        efield8_z[index] = 3;
        }*/
        
        //cout << u <<" "<<q<< endl;
        
        
        ///MOGOČE TO NI USTREZNO Z MOJIM PRIMEROM DA SO 0 POSTAVLJENI
        /*foriginal[0] -= x_force*unwrap[0] + y_force*unwrap[1] +z_force*unwrap[2];
        foriginal[1] += f[index][0];
        foriginal[2] += f[index][1];
        foriginal[3] += f[index][2];
        f[index][0] += x_force; // applied force from script!
        f[index][1] += y_force;
        f[index][2] += z_force;*/
        /*cout <<"u_x u_y u_z"<<u_x<<" "<<u_y<<" "<<u_z<<endl;
        cout <<"f_[i][0] f_[i][1] f_[i][2]"<<f[i][0]<<" "<<f[i][1]<<" "<<f[i][2]<<endl;
        cout <<"x_force y_force  z_force "<<x_force<<" "<<y_force<<" "<<z_force<<endl;
         cout <<"q"<<charg<<endl;
         cout<<"type"<<type[i]<<endl;*/
        //cout <<x[i][0]<<" "<<x[i][1]<<" "<<x[i][2]<<endl;
       
       /* if (evflag) {
          v[0] = x_force * unwrap[0];
          v[1] = y_force * unwrap[1];
          v[2] = z_force * unwrap[2];
          v[3] = x_force * unwrap[1];
          v[4] = x_force * unwrap[2];
          v[5] = z_force * unwrap[2];
          v_tally(index,v);
        }
      */

  // variable force, wrap with clear/add
  // potential energy = evar if defined, else 0.0
  // wrap with clear/add


/* ---------------------------------------------------------------------- */

//void FixAddForceEma::post_force_respa(int vflag, int ilevel, int /*iloop*/)
/*
{
  if (ilevel == ilevel_respa) post_force(vflag);
}
*/
/* ---------------------------------------------------------------------- */
/*
void FixAddForceEma::min_post_force(int vflag)
{
  post_force(vflag);
}
*/
/* ----------------------------------------------------------------------
   potential energy of added force
------------------------------------------------------------------------- */
/*
double FixAddForceEma::compute_scalar()
{
  // only sum across procs one time

  if (force_flag == 0) {
    MPI_Allreduce(foriginal,foriginal_all,4,MPI_DOUBLE,MPI_SUM,world);
    force_flag = 1;
  }
  return foriginal_all[0];
}
*/
/* ----------------------------------------------------------------------
   return components of total force on fix group before force was changed
------------------------------------------------------------------------- */
/*
double FixAddForceEma::compute_vector(int n)
{
  // only sum across procs one time

  if (force_flag == 0) {
    MPI_Allreduce(foriginal,foriginal_all,4,MPI_DOUBLE,MPI_SUM,world);
    force_flag = 1;
  }
  return foriginal_all[n+1];
}
*/
/* ----------------------------------------------------------------------
   memory usage of local atom-based array
------------------------------------------------------------------------- */
/*
double FixAddForceEma::memory_usage()
{
  double bytes = 0.0;
  if (varflag == ATOM) bytes = maxatom*4 * sizeof(double);
  return bytes;
}
*/
