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

/* ----------------------------------------------------------------------
   Contributing author: Anders Johansson (Harvard)
------------------------------------------------------------------------- */

#include <pair_nequip.h>
#include "atom.h"
#include "comm.h"
#include "domain.h"
#include "error.h"
#include "force.h"
#include "memory.h"
#include "neigh_list.h"
#include "neigh_request.h"
#include "neighbor.h"
#include "potential_file_reader.h"
#include "tokenizer.h"

#include <cmath>
#include <cstring>
#include <numeric>
#include <cassert>
#include <iostream>
#include <sstream>
#include <string>
#include <torch/torch.h>
#include <torch/script.h>
#include <torch/csrc/jit/runtime/graph_executor.h>
//#include <c10/cuda/CUDACachingAllocator.h>


// We have to do a backward compatability hack for <1.10
// https://discuss.pytorch.org/t/how-to-check-libtorch-version/77709/4
// Basically, the check in torch::jit::freeze
// (see https://github.com/pytorch/pytorch/blob/dfbd030854359207cb3040b864614affeace11ce/torch/csrc/jit/api/module.cpp#L479)
// is wrong, and we have ro "reimplement" the function
// to get around that...
// it's broken in 1.8 and 1.9
// BUT the internal logic in the function is wrong in 1.10
// So we only use torch::jit::freeze in >=1.11
#if (TORCH_VERSION_MAJOR == 1 && TORCH_VERSION_MINOR <= 10)
  #define DO_TORCH_FREEZE_HACK
  // For the hack, need more headers:
  #include <torch/csrc/jit/passes/freeze_module.h>
  #include <torch/csrc/jit/passes/frozen_conv_add_relu_fusion.h>
  #include <torch/csrc/jit/passes/frozen_graph_optimizations.h>
  #include <torch/csrc/jit/passes/frozen_ops_to_mkldnn.h>
#endif


using namespace LAMMPS_NS;

PairNEQUIP::PairNEQUIP(LAMMPS *lmp) : Pair(lmp) {
  restartinfo = 0;
  manybody_flag = 1;

  if(torch::cuda::is_available()){
    device = torch::kCUDA;
  }
  else {
    device = torch::kCPU;
  }
  std::cout << "NEQUIP is using device " << device << "\n";
  

  if(const char* env_p = std::getenv("NEQUIP_DEBUG")){
    std::cout << "PairNEQUIP is in DEBUG mode, since NEQUIP_DEBUG is in env\n";
    debug_mode = 1;
  }
}

PairNEQUIP::~PairNEQUIP(){
  if (allocated) {
    memory->destroy(setflag);
    memory->destroy(cutsq);
    memory->destroy(type_mapper);
  }
}

void PairNEQUIP::init_style(){
  if (atom->tag_enable == 0)
    error->all(FLERR,"Pair style NEQUIP requires atom IDs");

  // need a full neighbor list
  int irequest = neighbor->request(this,instance_me);
  neighbor->requests[irequest]->half = 0;
  neighbor->requests[irequest]->full = 1;

  // TODO: probably also
  neighbor->requests[irequest]->ghost = 0;

  // TODO: I think Newton should be off, enforce this.
  // The network should just directly compute the total forces
  // on the "real" atoms, with no need for reverse "communication".
  // May not matter, since f[j] will be 0 for the ghost atoms anyways.
  if (force->newton_pair == 1)
    error->all(FLERR,"Pair style NEQUIP requires newton pair off");
}

double PairNEQUIP::init_one(int i, int j)
{
  return cutoff;
}

void PairNEQUIP::allocate()
{
  allocated = 1;
  int n = atom->ntypes;
  
  memory->create(setflag,n+1,n+1,"pair:setflag");
  memory->create(cutsq,n+1,n+1,"pair:cutsq");
  memory->create(type_mapper, n+1, "pair:type_mapper");

}

void PairNEQUIP::settings(int narg, char ** /*arg*/) {
  // "flare" should be the only word after "pair_style" in the input file.
  if (narg > 0)
    error->all(FLERR, "Illegal pair_style command");
}

void PairNEQUIP::coeff(int narg, char **arg) {

  if (!allocated)
    allocate();

  int ntypes = atom->ntypes;

  // Should be exactly 3 arguments following "pair_coeff" in the input file.
  if (narg != (3+ntypes))
    error->all(FLERR, "Incorrect args for pair coefficients");

  // Ensure I,J args are "* *".
  if (strcmp(arg[0], "*") != 0 || strcmp(arg[1], "*") != 0)
    error->all(FLERR, "Incorrect args for pair coefficients");

  for (int i = 1; i <= ntypes; i++)
    for (int j = i; j <= ntypes; j++)
      setflag[i][j] = 0;

  // Parse the definition of each atom type
  char **elements = new char*[ntypes+1];
  for (int i = 1; i <= ntypes; i++){
      elements[i] = new char [strlen(arg[i+2])+1];
      strcpy(elements[i], arg[i+2]);
      if (screen) fprintf(screen, "NequIP Coeff: type %d is element %s\n", i, elements[i]);
  }
 
  // Initiate type mapper
  for (int i = 1; i<= ntypes; i++){
      type_mapper[i] = -1;
  }

  std::cout << "Loading model from " << arg[2] << "\n";

  std::unordered_map<std::string, std::string> metadata = {
    {"config", ""},
    {"nequip_version", ""},
    {"r_max", ""},
    {"n_species", ""},
    {"type_names", ""},
    {"_jit_bailout_depth", ""},
    {"_jit_fusion_strategy", ""},
    {"allow_tf32", ""}
  };
  model = torch::jit::load(std::string(arg[2]), device, metadata);
  model.eval();

  // Check if model is a NequIP model
  if (metadata["nequip_version"].empty()) {
    error->all(FLERR, "The indicated TorchScript file does not appear to be a deployed NequIP model; did you forget to run `nequip-deploy`?");
  }

  // If the model is not already frozen, we should freeze it:
  // This is the check used by PyTorch: https://github.com/pytorch/pytorch/blob/master/torch/csrc/jit/api/module.cpp#L476
  if (model.hasattr("training")) {
    std::cout << "Freezing TorchScript model...\n";
    #ifdef DO_TORCH_FREEZE_HACK
      // Do the hack
      // Copied from the implementation of torch::jit::freeze,
      // except without the broken check
      // See https://github.com/pytorch/pytorch/blob/dfbd030854359207cb3040b864614affeace11ce/torch/csrc/jit/api/module.cpp
      bool optimize_numerics = true;  // the default
      // the {} is preserved_attrs
      auto out_mod = freeze_module(
        model, {}
      );
      // See 1.11 bugfix in https://github.com/pytorch/pytorch/pull/71436
      auto graph = out_mod.get_method("forward").graph();
      OptimizeFrozenGraph(graph, optimize_numerics);
      model = out_mod;
    #else
      // Do it normally
      model = torch::jit::freeze(model);
    #endif
  }

  #if (TORCH_VERSION_MAJOR == 1 && TORCH_VERSION_MINOR <= 10)
    // Set JIT bailout to avoid long recompilations for many steps
    size_t jit_bailout_depth;
    if (metadata["_jit_bailout_depth"].empty()) {
      // This is the default used in the Python code
      jit_bailout_depth = 2;
    } else {
      jit_bailout_depth = std::stoi(metadata["_jit_bailout_depth"]);
    }
    torch::jit::getBailoutDepth() = jit_bailout_depth;
  #else
    // In PyTorch >=1.11, this is now set_fusion_strategy
    torch::jit::FusionStrategy strategy;
    if (metadata["_jit_fusion_strategy"].empty()) {
      // This is the default used in the Python code
      strategy = {{torch::jit::FusionBehavior::DYNAMIC, 3}};
    } else {
      std::stringstream strat_stream(metadata["_jit_fusion_strategy"]);
      std::string fusion_type, fusion_depth;
      while(std::getline(strat_stream, fusion_type, ',')) {
        std::getline(strat_stream, fusion_depth, ';');
        strategy.push_back({fusion_type == "STATIC" ? torch::jit::FusionBehavior::STATIC : torch::jit::FusionBehavior::DYNAMIC, std::stoi(fusion_depth)});
      }
    }
    torch::jit::setFusionStrategy(strategy);
  #endif

  // Set whether to allow TF32:
  bool allow_tf32;
  if (metadata["allow_tf32"].empty()) {
    // Better safe than sorry
    allow_tf32 = false;
  } else {
    // It gets saved as an int 0/1
    allow_tf32 = std::stoi(metadata["allow_tf32"]);
  }
  // See https://pytorch.org/docs/stable/notes/cuda.html
  at::globalContext().setAllowTF32CuBLAS(allow_tf32);
  at::globalContext().setAllowTF32CuDNN(allow_tf32);

  // std::cout << "Information from model: " << metadata.size() << " key-value pairs\n";
  // for( const auto& n : metadata ) {
  //   std::cout << "Key:[" << n.first << "] Value:[" << n.second << "]\n";
  // }

  cutoff = std::stod(metadata["r_max"]);

  // match the type names in the pair_coeff to the metadata
  // to construct a type mapper from LAMMPS type to NequIP atom_types
  int n_species = std::stod(metadata["n_species"]);
  std::stringstream ss;
  ss << metadata["type_names"];
  for (int i = 0; i < n_species; i++){
      char ele[100];
      ss >> ele;
      for (int itype = 1; itype <= ntypes; itype++)
          if (strcmp(elements[itype], ele) == 0)
              type_mapper[itype] = i;
   
  }

  // set setflag i,j for type pairs where both are mapped to elements
  for (int i = 1; i <= ntypes; i++){
    for (int j = i; j <= ntypes; j++){
        if ((type_mapper[i] >= 0) && (type_mapper[j] >= 0)){
            setflag[i][j] = 1;
        std::cout<<"typemapper i "<<type_mapper[i]<<" typemapper j "<<type_mapper[j]<< " setflag[i][j] "<<setflag[i][j]<<std::endl;}}} 
        

  if (elements){
      for (int i=1; i<ntypes; i++)
          if (elements[i]) delete [] elements[i];
      delete [] elements;
  }

}

// Force and energy computation
void PairNEQUIP::compute(int eflag, int vflag){
  ev_init(eflag, vflag);

  // Get info from lammps:
  
  // Atom positions, including ghost atoms
  double **x = atom->x;
  // Atom forces
  double **f = atom->f;
  // Atom IDs, unique, reproducible, the "real" indices
  // Probably 1-based
  tagint *tag = atom->tag;
  // Atom types, 1-based
  int *type = atom->type;
  // Number of local/real atoms
  int nlocal = atom->nlocal;
  //std::cout << "nlocal..."<<nlocal<<std::endl;
  // Whether Newton is on (i.e. reverse "communication" of forces on ghost atoms).
  int newton_pair = force->newton_pair;
  // Should probably be off.
  if (newton_pair==1)
    error->all(FLERR,"Pair style NEQUIP requires 'newton off'");
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
  // Number of local/real atoms
  int inum = list->inum;
  //std::cout << "inum..."<<inum<<std::endl; //je stevilo neigh list, ce je pair style hybirde ni nujno enak kot nlocal, nlocal so vsi localni atomi, inum st neigh list za ta pair style - ce gleda tipe samo Na Cl, potem tu samo 323 
  assert(inum==nlocal); // This should be true, if my understanding is correct
  // Number of ghost atoms
  int nghost = list->gnum;
  // Total number of atoms
  int ntotal = inum + nghost; //POPRAVILA inum + nghost
 // std::cout << "ntotal..."<<ntotal<<std::endl;
 // std::cout << "nghost..."<<nghost<<std::endl;
  // Mapping from neigh list ordering to x/f ordering
  int *ilist = list->ilist;
  // Number of neighbors per atom
  int *numneigh = list->numneigh;
  // Neighbor list per atom
  int **firstneigh = list->firstneigh;
  //std::cout << "numneigh..."<<numneigh[0]<<std::endl;
  //std::cout << "numneigh+ntotal..."<<numneigh[nlocal-1]<<std::endl;
  //std::cout << "numneighP..."<<numneigh[280]<<std::endl;
  // Total number of bonds (sum of number of neighbors)
  int nedges = std::accumulate(numneigh, numneigh+ntotal, 0);
  //std::cout << "nedges..."<<nedges<<std::endl;
  torch::Tensor pos_tensor = torch::zeros({inum, 3});
  torch::Tensor EF_tensor = torch::zeros({inum, 3});
  torch::Tensor EF1_tensor = torch::zeros({inum, 3});
  torch::Tensor EF2_tensor = torch::zeros({inum, 3});
  torch::Tensor EF3_tensor = torch::zeros({inum, 3});
  torch::Tensor EF4_tensor = torch::zeros({inum, 3});
  torch::Tensor EF5_tensor = torch::zeros({inum, 3});
  torch::Tensor EF6_tensor = torch::zeros({inum, 3});
  torch::Tensor EF7_tensor = torch::zeros({inum, 3});
  torch::Tensor EF8_tensor = torch::zeros({inum, 3});
  //std::cout << "postensor..."<<pos_tensor<<std::endl;
  torch::Tensor tag2type_tensor = torch::zeros({inum}, torch::TensorOptions().dtype(torch::kInt64));
  //std::cout << "tag2type_tensor.."<<std::endl;
  torch::Tensor periodic_shift_tensor = torch::zeros({3});
  //std::cout << "periodic_shift_tensor.."<<periodic_shift_tensor<<std::endl;
  torch::Tensor cell_tensor = torch::zeros({3,3});
  //std::cout << "cell_tensor.."<<cell_tensor<<std::endl;
 
  
  auto pos = pos_tensor.accessor<float, 2>();
  auto EF =  EF_tensor.accessor<float, 2>();
  auto EF1 = EF1_tensor.accessor<float, 2>();
  auto EF2 = EF2_tensor.accessor<float, 2>();
  auto EF3 = EF3_tensor.accessor<float, 2>();
  auto EF4 = EF4_tensor.accessor<float, 2>();
  auto EF5 = EF5_tensor.accessor<float, 2>();
  auto EF6 = EF6_tensor.accessor<float, 2>();
  auto EF7 = EF7_tensor.accessor<float, 2>();
  auto EF8 = EF8_tensor.accessor<float, 2>();
  //std::cout << "auto pos.."<<std::endl;
  long edges[2*nedges];
  //std::cout << "long edges.."<<std::endl;
  float edge_cell_shifts[3*nedges];
  //std::cout << "edge_cell_shifts.."<<std::endl;
  auto tag2type = tag2type_tensor.accessor<long, 1>();
  auto periodic_shift = periodic_shift_tensor.accessor<float, 1>();
  auto cell = cell_tensor.accessor<float,2>();
  //std::cout << "cell_tensor.."<<cell<<std::endl;
  //std::cout << "end auto.."<<std::endl;
  // Inverse mapping from tag to "real" atom index
  std::vector<int> tag2i(inum);

  // Loop over real atoms to store tags, types and positions
  // std::cout << "inum..."<<inum<<std::endl;
   int count_Na = 0;
   int count_Cl = 0;
  for(int ii = 0; ii < inum; ii++){
    int i = ilist[ii]; //avtomaticno preskoci index i tako da gre skozi 323 (0-322) krat ampak dobi vse info
    int itag = tag[i];
    int itype = type[i];
   // if (type[i] != 1 && type[i] != 2){
     //   continue;}
    if (itype == 2){count_Na += 1;};
    if (itype == 1){count_Cl += 1;};
    // Inverse mapping from tag to x/f atom index
    tag2i[itag-1] = i; // tag is probably 1-based
    
    tag2type[itag-1] = type_mapper[itype];
    pos[itag-1][0] = x[i][0];
    pos[itag-1][1] = x[i][1];
    pos[itag-1][2] = x[i][2];
    
    EF[itag-1][0] = efield_x[i];
    EF[itag-1][1] =efield_y[i];
    EF[itag-1][2] = efield_z[i];
    
    EF1[itag-1][0] = efield1_x[i];
    EF1[itag-1][1] = efield1_y[i];
    EF1[itag-1][2] = efield1_z[i];
    
    EF2[itag-1][0] = efield2_x[i];
    EF2[itag-1][1] = efield2_y[i];
    EF2[itag-1][2] = efield2_z[i];
    
    EF3[itag-1][0] = efield3_x[i];
    EF3[itag-1][1] = efield3_y[i];
    EF3[itag-1][2] =efield3_z[i];
    
    EF4[itag-1][0] = efield4_x[i];
    EF4[itag-1][1] = efield4_y[i];
    EF4[itag-1][2] = efield4_z[i];
    
    EF5[itag-1][0] = efield5_x[i];
    EF5[itag-1][1] = efield5_y[i];
    EF5[itag-1][2] = efield5_z[i];
    
    EF6[itag-1][0] = efield6_x[i];
    EF6[itag-1][1] = efield6_y[i];
    EF6[itag-1][2] = efield6_z[i];
    
    EF7[itag-1][0] = efield7_x[i];
    EF7[itag-1][1] = efield7_y[i];
    EF7[itag-1][2] = efield7_z[i];
    
    EF8[itag-1][0] = efield8_x[i];
    EF8[itag-1][1] = efield8_y[i];
    EF8[itag-1][2] = efield8_z[i];
    
    
   // std::cout << "ii  "<<ii<<" i "<<i<<" itag  "<<itag<<" itype "<<itype<<" inumtype  "<<type[ii]<<" inum tag "<<tag[ii]<<std::endl;
   // std::cout << "EF[i][0]  "<<EF[itag-1][0]<<" pos[itag-1][0] "<<pos[itag-1][0]<<"x[j][1]  "<<x[i][1]<<" pos[itag-1][1] "<<pos[itag-1][1]<<"x[i][2]  "<<x[i][2]<<" pos[itag-1][2] "<<pos[itag-1][2]<<std::endl;
    /*std::cout << "EF[i][0]  "<<EF[itag-1][0]<<"  EF[j][1]  "<<EF[itag-1][1]<<" EF[i][2]  "<<EF[itag-1][2]<<" EF1[i][0]  "<<EF1[itag-1][0]<<"  EF1[j][1]  "<<EF1[itag-1][1]<<" EF1[i][2]  "<<EF1[itag-1][2]<<"EF2[i][0]  "<<EF2[itag-1][0]<<"  EF2[j][1]  "<<EF2[itag-1][1]<<" EF2[i][2]  "<<EF2[itag-1][2]<<"EF3[i][0]  "<<EF3[itag-1][0]<<"  EF3[j][1]  "<<EF3[itag-1][1]<<" EF3[i][2]  "<<EF3[itag-1][2]<<"EF4[i][0]  "<<EF4[itag-1][0]<<"  EF4[j][1]  "<<EF4[itag-1][1]<<" EF4[i][2]  "<<EF4[itag-1][2]<<"EF5[i][0]  "<<EF5[itag-1][0]<<"  EF5[j][1]  "<<EF5[itag-1][1]<<" EF5[i][2]  "<<EF5[itag-1][2]<<"EF6[i][0]  "<<EF6[itag-1][0]<<"  EF6[j][1]  "<<EF6[itag-1][1]<<" EF6[i][2]  "<<EF6[itag-1][2]<<"EF7[i][0]  "<<EF7[itag-1][0]<<"  EF7[j][1]  "<<EF7[itag-1][1]<<" EF7[i][2]  "<<EF7[itag-1][2]<<"EF8[i][0]  "<<EF8[itag-1][0]<<"  EF8[j][1]  "<<EF8[itag-1][1]<<" EF8[i][2]  "<<EF8[itag-1][2]<<std::endl;*/
  }
  //std::cout << "x[i][0]  "<<x[33][0]<<" pos[itag-1][0] "<<pos[323][0]<<"x[j][1]  "<<x[33][1]<<" pos[jtag-1][1] "<<pos[323][1]<<"x[j][2]  "<<x[323][2]<<" pos[jtag-1][2] "<<pos[323][2]<<std::endl;
  //std::cout << "count_Na  "<<count_Na<<std::endl;
  //std::cout << "count_Cl  "<<count_Cl<<std::endl;
  // Get cell
  cell[0][0] = domain->boxhi[0] - domain->boxlo[0];

  cell[1][0] = domain->xy;
  cell[1][1] = domain->boxhi[1] - domain->boxlo[1];

  cell[2][0] = domain->xz;
  cell[2][1] = domain->yz;
  cell[2][2] = domain->boxhi[2] - domain->boxlo[2];
  /*std::cout<<cell[0][0]<<std::endl;
  std::cout<<cell[1][0]<<std::endl;
  std::cout<<cell[1][1]<<std::endl;
  std::cout<<cell[2][0]<<std::endl;
  std::cout<<cell[2][1]<<std::endl;
  std::cout<<cell[2][2]<<std::endl;*/
  auto cell_inv = cell_tensor.inverse().transpose(0,1);
  
  
  //exit(EXIT_FAILURE);
  //std::cout << "end cell"<<std::endl;
  /*
  std::cout << "cell: " << cell_tensor << "\n";
  std::cout << "tag2i: " << "\n";
  for(int itag = 0; itag < inum; itag++){
    std::cout << tag2i[itag] << " ";
  }
  std::cout << std::endl;
  */

 // auto cell_inv = cell_tensor.inverse().transpose(0,1);

  // Loop over atoms and neighbors,
  // store edges and _cell_shifts
  // ii follows the order of the neighbor lists,
  // i follows the order of x, f, etc.
  int edge_counter = 0;
  if (debug_mode) printf("NEQUIP edges: i j xi[:] xj[:] cell_shift[:] rij\n");
  //std::cout << "negin nlocal loop"<<std::endl;
  int cout_Na = 0;
  int cout_Cl = 0;
  for(int ii = 0; ii < inum; ii++){ //avtomaticno preskoci index i tako da gre skozi 323 (0-322) krat ampak dobi vse info
   // std::cout << "ii  "<<ii<<std::endl;
   // std::cout << "ilist[323]  "<<ilist[323]<<" tag[323] "<<tag[323]<<std::endl;
    int i = ilist[ii];
    int itag = tag[i];
    int itype = type[i];
    //if (type[i] != 1 && type[i] != 2){
     //   continue;}
    //if (itype == 2){cout_Na +=1 ;}
   // if (itype == 1){cout_Cl +=1 ;}
   // std::cout << "cout_Na  "<<cout_Na<<" cout_Cl "<<cout_Cl<<std::endl;
    //std::cout << "ii  "<<ii<<" i inum "<<i<<std::endl;
    //std::cout << "i inum "<<i<<" ii local "<<ii<<" type ii  "<<type[ii]<<" type i  "<<type[i]<<" tag i  "<<tag[i]<<" tag ii  "<<tag[ii]<<" numneigh[0] "<<numneigh[ii]<<std::endl;
    //std::cout << "x[i][0]  "<<x[i][0]<<" pos[itag-1][0] "<<pos[itag-1][0]<<"x[j][1]  "<<x[i][1]<<" pos[itag-1][1] "<<pos[itag-1][1]<<"x[i][2]  "<<x[i][2]<<" pos[itag-1][2] "<<pos[itag-1][2]<<std::endl;
    int jnum = numneigh[i];
    int *jlist = firstneigh[i];
    for(int jj = 0; jj < jnum; jj++){
       
      int j = jlist[jj];
      j &= NEIGHMASK;
      int jtag = tag[j];
      int jtype = type[j];
     // if (type[j] != 1 && type[j] != 2){
      //  continue;}
      //std::cout << "j inum "<<j<<" jj total "<<jj<<" type jj  "<<type[jj]<<" type j "<<type[j]<<" tag j  "<<tag[j]<<" tag jj  "<<tag[jj]<<std::endl;
     
      // TODO: check sign
     //  std::cout << "pred periodic  "<<i<<std::endl;
      periodic_shift[0] = x[j][0] - pos[jtag-1][0]; //ghost atome premakne na pozicijo tega globalnega taga
      periodic_shift[1] = x[j][1] - pos[jtag-1][1];
      periodic_shift[2] = x[j][2] - pos[jtag-1][2];
     // std::cout << "x[j][0]  "<<x[j][0]<<" pos[jtag-1][0] "<<pos[jtag-1][0]<<"x[j][1]  "<<x[j][1]<<" pos[jtag-1][1] "<<pos[jtag-1][1]<<"x[j][2]  "<<x[j][2]<<" pos[jtag-1][2] "<<pos[jtag-2][2]<<std::endl;
      //std::cout << "EF[i][0]  "<<EF[jtag-1][0]<<"  EF[j][1]  "<<EF[jtag-1][1]<<" EF[i][2]  "<<EF[jtag-1][2]<<" EF8[i][2]  "<<EF8[jtag-1][2]<<std::endl;
      double dx = x[i][0] - x[j][0];
      double dy = x[i][1] - x[j][1];
      double dz = x[i][2] - x[j][2];
     //   std::cout << "po dx  "<<i<<j<<"  "<<jtype<<"  jtag-1  "<<jtag-1<<" "<<periodic_shift[2]<<std::endl;
     
      double rsq = dx*dx + dy*dy + dz*dz;
     //   std::cout << "predcutoff  "<<rsq<<std::endl;
      //  std::cout << "cutoff  "<<cutoff*cutoff<<std::endl;
      if (rsq < cutoff*cutoff){
         //   std::cout << "vcutoff  "<<itag - 1<<"  "<<jtag - 1<<std::endl;
          // std::cout<<cell<<std::endl;
           //std::cout<<periodic_shift_tensor<<std::endl;
          torch::Tensor cell_shift_tensor = cell_inv.matmul(periodic_shift_tensor); //matmul: pomnozitev dveh matrik cell=cell_tenosr, cell_inv =inv(cell) in peridic shift = periodic shift tensor
          // std::cout << "cellshifttensor  "<<cell_shift_tensor<<std::endl;
         
          auto cell_shift = cell_shift_tensor.accessor<float, 1>(); //is 1-dimensional and holds floats TU TEZAVA GHOST 616 TAG 33
          // std::cout << " autocellshifttensor  "<<std::endl;
          float * e_vec = &edge_cell_shifts[edge_counter*3];
           //std::cout << "float e_vec  "<<itag - 1<<std::endl;
          e_vec[0] = std::round(cell_shift[0]);
          e_vec[1] = std::round(cell_shift[1]);
          e_vec[2] = std::round(cell_shift[2]);
          // std::cout << "cell shift: " << cell_shift_tensor << "\n";

          // TODO: double check order
          //  std::cout << "itag  "<<itag - 1<<std::endl;
          edges[edge_counter*2] = itag - 1; // std::cout << "itag  "<<itag - 1<<std::endl; // tag is probably 1-based
          edges[edge_counter*2+1] = jtag - 1; // std::cout << "jtag  "<<jtag - 1<<std::endl; // tag is probably 1-based
          edge_counter++;

          if (debug_mode){
              printf("%d %d %.10g %.10g %.10g %.10g %.10g %.10g %.10g %.10g %.10g %.10g\n", itag-1, jtag-1,
                pos[itag-1][0],pos[itag-1][1],pos[itag-1][2],pos[jtag-1][0],pos[jtag-1][1],pos[jtag-1][2],
                e_vec[0],e_vec[1],e_vec[2],sqrt(rsq));
          }  // std::cout << "after debug  "<<i<<j<<std::endl;

      }  // std::cout << "after cutoff  "<<i<<j<<std::endl;
    }  // std::cout << "after jloop  "<<i<<std::endl;
  }
    //std::cout << "end nlocal loop"<<std::endl;
  if (debug_mode) printf("end NEQUIP edges\n");

  // shorten the list before sending to nequip
  torch::Tensor edges_tensor = torch::zeros({2,edge_counter}, torch::TensorOptions().dtype(torch::kInt64));
  torch::Tensor edge_cell_shifts_tensor = torch::zeros({edge_counter,3});
  auto new_edges = edges_tensor.accessor<long, 2>();
  auto new_edge_cell_shifts = edge_cell_shifts_tensor.accessor<float, 2>();
  for (int i=0; i<edge_counter; i++){

      long *e=&edges[i*2];
      new_edges[0][i] = e[0];
      new_edges[1][i] = e[1];

      float *ev = &edge_cell_shifts[i*3];
      new_edge_cell_shifts[i][0] = ev[0];
      new_edge_cell_shifts[i][1] = ev[1];
      new_edge_cell_shifts[i][2] = ev[2];
  }

  
  c10::Dict<std::string, torch::Tensor> input;
  input.insert("pos", pos_tensor.to(device));
  input.insert("EF",EF_tensor.to(device));
  input.insert("EF1",EF1_tensor.to(device));
  input.insert("EF2",EF2_tensor.to(device));
  input.insert("EF3",EF3_tensor.to(device));
  input.insert("EF4",EF4_tensor.to(device));
  input.insert("EF5",EF5_tensor.to(device));
  input.insert("EF6",EF6_tensor.to(device));
  input.insert("EF7",EF7_tensor.to(device));
  input.insert("EF8",EF8_tensor.to(device));
  input.insert("edge_index", edges_tensor.to(device));
  input.insert("edge_cell_shift", edge_cell_shifts_tensor.to(device));
  input.insert("cell", cell_tensor.to(device));
  input.insert("atom_types", tag2type_tensor.to(device));
  std::vector<torch::IValue> input_vector(1, input);

  if(debug_mode){
    std::cout << "NequIP model input:\n";
    std::cout << "pos:\n" << pos_tensor << "\n";
    std::cout << "edge_index:\n" << edges_tensor << "\n";
    std::cout << "edge_cell_shifts:\n" << edge_cell_shifts_tensor << "\n";
    std::cout << "cell:\n" << cell_tensor << "\n";
    std::cout << "atom_types:\n" << tag2type_tensor << "\n";
  }


  auto output = model.forward(input_vector).toGenericDict();

  torch::Tensor forces_tensor = output.at("forces").toTensor().cpu();
  auto forces = forces_tensor.accessor<float, 2>();

  torch::Tensor total_energy_tensor = output.at("total_energy").toTensor().cpu();

  // store the total energy where LAMMPS wants it
  eng_vdwl = total_energy_tensor.data_ptr<float>()[0];

  torch::Tensor atomic_energy_tensor = output.at("atomic_energy").toTensor().cpu();
  auto atomic_energies = atomic_energy_tensor.accessor<float, 2>();
  float atomic_energy_sum = atomic_energy_tensor.sum().data_ptr<float>()[0];

  if(debug_mode){
    std::cout << "NequIP model output:\n";
    std::cout << "forces: " << forces_tensor << "\n";
    std::cout << "total_energy: " << total_energy_tensor << "\n";
    std::cout << "atomic_energy: " << atomic_energy_tensor << "\n";
  }

  //std::cout << "atomic energy sum: " << atomic_energy_sum << std::endl;
  //std::cout << "Total energy: " << total_energy_tensor << "\n";
  //std::cout << "atomic energy shape: " << atomic_energy_tensor.sizes()[0] << "," << atomic_energy_tensor.sizes()[1] << std::endl;
  //std::cout << "atomic energies: " << atomic_energy_tensor << std::endl;

  // Write forces and per-atom energies (0-based tags here)
  for(int itag = 0; itag < inum; itag++){
    int i = tag2i[itag];
    f[i][0] = forces[itag][0];
    f[i][1] = forces[itag][1];
    f[i][2] = forces[itag][2];
    if (eflag_atom) eatom[i] = atomic_energies[itag][0];
    //printf("%d %d %g %g %g %g %g %g\n", i, type[i], pos[itag][0], pos[itag][1], pos[itag][2], f[i][0], f[i][1], f[i][2]);
  }

  // TODO: Virial stuff? (If there even is a pairwise force concept here)

  // TODO: Performance: Depending on how the graph network works, using tags for edges may lead to shitty memory access patterns and performance.
  // It may be better to first create tag2i as a separate loop, then set edges[edge_counter][:] = (i, tag2i[jtag]).
  // Then use forces(i,0) instead of forces(itag,0).
  // Or just sort the edges somehow.

  /*
  if(device.is_cuda()){
    //torch::cuda::empty_cache();
    c10::cuda::CUDACachingAllocator::emptyCache();
  }
  */
}
