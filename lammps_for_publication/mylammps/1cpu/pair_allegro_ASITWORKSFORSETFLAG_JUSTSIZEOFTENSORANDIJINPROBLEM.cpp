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

#include <pair_allegro.h>
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

#include <algorithm>
#include <vector>
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

// TODO: Only if MPI is available
#include <mpi.h>



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

PairAllegro::PairAllegro(LAMMPS *lmp) : Pair(lmp) {
  restartinfo = 0;
  manybody_flag = 1;

  if(const char* env_p = std::getenv("ALLEGRO_DEBUG")){
    std::cout << "PairAllegro is in DEBUG mode, since ALLEGRO_DEBUG is in env\n";
    debug_mode = 1;
  }

  if(torch::cuda::is_available()){
    int deviceidx = -1;
    if(comm->nprocs > 1){
      MPI_Comm shmcomm;
      MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0,
          MPI_INFO_NULL, &shmcomm);
      int shmrank;
      MPI_Comm_rank(shmcomm, &shmrank);
      deviceidx = shmrank;
    }
    if(deviceidx >= 0) {
      int devicecount = torch::cuda::device_count();
      if(deviceidx >= devicecount) {
        if(debug_mode) {
          // To allow testing multi-rank calls, we need to support multiple ranks with one GPU
          std::cerr << "WARNING (Allegro): my rank (" << deviceidx << ") is bigger than the number of visible devices (" << devicecount << "), wrapping around to use device " << deviceidx % devicecount << " again!!!";
          deviceidx = deviceidx % devicecount;
        }
        else {
          // Otherwise, more ranks than GPUs is an error
          std::cerr << "ERROR (Allegro): my rank (" << deviceidx << ") is bigger than the number of visible devices (" << devicecount << ")!!!";
          error->all(FLERR,"pair_allegro: mismatch between number of ranks and number of available GPUs");
        }
      }
    }
    device = c10::Device(torch::kCUDA,deviceidx);
  }
  else {
    device = torch::kCPU;
  }
  std::cout << "Allegro is using device " << device << "\n";
}

PairAllegro::~PairAllegro(){
  if (allocated) {
    memory->destroy(setflag);
    memory->destroy(cutsq);
   // memory->destroy(type_mapper); //dodala
  }
}

void PairAllegro::init_style(){
  if (atom->tag_enable == 0)
    error->all(FLERR,"Pair style Allegro requires atom IDs");

  // need a full neighbor list
  int irequest = neighbor->request(this,instance_me);
  neighbor->requests[irequest]->half = 0;
  neighbor->requests[irequest]->full = 1;

  neighbor->requests[irequest]->ghost = 1;

 /* if (force->newton_pair == 0)
    error->all(FLERR,"Pair style Allegro requires newton pair on");*/
}

double PairAllegro::init_one(int i, int j)
{
  return cutoff;
}

void PairAllegro::allocate()
{
  allocated = 1;
  int n = atom->ntypes;

  memory->create(setflag,n+1,n+1,"pair:setflag");
  memory->create(cutsq,n+1,n+1,"pair:cutsq");
  //memory->create(type_mapper, n+1, "pair:type_mapper"); //DODALA
}

void PairAllegro::settings(int narg, char ** /*arg*/) {
  // "allegro" should be the only word after "pair_style" in the input file.
  if (narg > 0)
    error->all(FLERR, "Illegal pair_style command, too many arguments");
}

void PairAllegro::coeff(int narg, char **arg) {
  if (!allocated)
    allocate();

  int ntypes = atom->ntypes;

  // Should be exactly 3 arguments following "pair_coeff" in the input file.
  if (narg != (3+ntypes))
    error->all(FLERR, "Incorrect args for pair coefficients, should be * * <model>.pth <type1> <type2> ... <typen>");

  // Ensure I,J args are "* *".
  if (strcmp(arg[0], "*") != 0 || strcmp(arg[1], "*") != 0)
    error->all(FLERR, "Incorrect args for pair coefficients");

  for (int i = 1; i <= ntypes; i++)
    for (int j = i; j <= ntypes; j++)
      setflag[i][j] = 0;

  std::vector<std::string> elements(ntypes);
  for(int i = 0; i < ntypes; i++){
    elements[i] = arg[i+1];}
  
  // Initiate type mapper
 /* for (int i = 1; i<= ntypes; i++){
      type_mapper[i] = -1;}*/
   //DODALA

  std::cout << "Allegro: Loading model from " << arg[2] << "\n";

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
    std::cout << "Allegro: Freezing TorchScript model...\n";
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

  // std::cout << "Allegro: Information from model: " << metadata.size() << " key-value pairs\n";
  // for( const auto& n : metadata ) {
  //   std::cout << "Key:[" << n.first << "] Value:[" << n.second << "]\n";
  // }

  cutoff = std::stod(metadata["r_max"]);

  //TODO: This
 /* type_mapper.resize(ntypes, -1);
  std::stringstream ss;
  int n_species = std::stod(metadata["n_species"]);
  ss << metadata["type_names"];
  std::cout << "Type mapping:" << "\n";
  std::cout << "Allegro type | Allegro name | LAMMPS type | LAMMPS name" << "\n";
  for (int i = 0; i < n_species; i++){
    std::string ele;
    ss >> ele;
    for (int itype = 1; itype <= ntypes; itype++){
      if (ele.compare(arg[itype + 3 - 1]) == 0){
        type_mapper[itype-1] = i;
        std::cout << i << " | " << ele << " | " << itype << " | " << arg[itype + 3 - 1] << "\n";
      }
    }
  }*/
  type_mapper.resize(ntypes,1); //imamo 4 pozcije CE HOCES DA SET FLAG DELA MORA TU BIT -1, NTOTAL PRAVI KOT MORA BITI BREZ BEAD = INUM -NGOST = STEVILO SAMO Na+Cl, povsot v kodi nlocal v inum, n total MORAS POPRAVITI NA INUM+NGOHST ne NLOCAL+NGHOST; PROBLEM JE KER SO INDEXI PRAVI AMPAK POS, EF, .. NISO PRAVE VELIKOSTI DA BI TE INDEXE POVEZALO S PRAVO POZICIJO
  /*for (int i = 1; i<= ntypes; i++){
      type_mapper[i] = -1; //pozicija 0 je 0, -1, -1. -1 
  }*/
  std::cout<<type_mapper[0]<<" type_mapper[0] "<<std::endl;
  std::cout<<type_mapper[1]<<" type_mapper[1] "<<std::endl;
  std::cout<<type_mapper[2]<<" type_mapper[2] "<<std::endl;
  //std::cout<<type_mapper[3]<<" type_mapper[3] "<<std::endl;
  std::cout<<type_mapper.size()<<" type_mapper.size() "<<std::endl;
  
  std::stringstream ss;
  int n_species = std::stod(metadata["n_species"]);
  ss << metadata["type_names"];
  std::cout << "Type mapping:" << "\n";
  std::cout << "Allegro type | Allegro name | LAMMPS type | LAMMPS name" << "\n";
  for (int i = 0; i < n_species; i++){
    std::string ele;
    ss >> ele;
    for (int itype = 1; itype <= ntypes; itype++){
      if (ele.compare(arg[itype + 3 - 1]) == 0){
        type_mapper[itype-1] = i;
        std::cout << i << " | " << ele << " | " << itype << " | " << arg[itype + 3 - 1] << "\n";
      }
    }
  }

  std::cout<<ntypes<<" ntypes "<<std::endl;
  std::cout<<type_mapper[0]<<" type_mapper[0] "<<std::endl;
  std::cout<<type_mapper[1]<<" type_mapper[1] "<<std::endl;
  std::cout<<type_mapper[2]<<" type_mapper[2] "<<std::endl;
  // set setflag i,j for type pairs where both are mapped to elements
   for (int i = 1; i <= ntypes; i++) {
    for (int j = i; j <= ntypes; j++) {
        if ((type_mapper[i-1] >= 0) && (type_mapper[j-1] >= 0)) {
            setflag[i][j] = 1;
        

        std::cout<<"typemapper i "<<type_mapper[i]<<" typemapper j "<<type_mapper[j]<< " setflag[i][j] "<<setflag[i][j]<<std::endl;}}}
        
 


  char *batchstr = std::getenv("BATCHSIZE");
  if (batchstr != NULL) {
    batch_size = std::atoi(batchstr);
  }

}

// Force and energy computation
void PairAllegro::compute(int eflag, int vflag){
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
  
  // Whether Newton is on (i.e. reverse "communication" of forces on ghost atoms).
  // Should be on.
  int newton_pair = force->newton_pair;

  // Number of local/real atoms
  int inum = list->inum;
  
  assert(inum==nlocal); // This should be true, if my understanding is correct
  // Number of ghost atoms
  int nghost = list->gnum;
  // Total number of atoms
  int ntotal = inum + nghost;
std::cout << "ntotal..."<<ntotal<<std::endl;
  std::cout << "nghost..."<<nghost<<std::endl;
  std::cout << "inum..."<<inum<<std::endl;
  std::cout << "nlocal..."<<nlocal<<std::endl;
  // Mapping from neigh list ordering to x/f ordering
  int *ilist = list->ilist;
  // Number of neighbors per atom
  int *numneigh = list->numneigh;
  // Neighbor list per atom
  int **firstneigh = list->firstneigh;


  // Total number of bonds (sum of number of neighbors)
  int nedges = 0;
    
  // Number of bonds per atom
  std::vector<int> neigh_per_atom(inum, 0);

#pragma omp parallel for reduction(+:nedges)
  for(int ii = 0; ii <inum; ii++){
    int i = ilist[ii];
    //std::cout<< " in neigh loop "<<std::endl;
    if (type[i] != 1 && type[i] != 2){
       continue;}
    if (type[i] == 3){std::cout<<"type bead in 1 i"<<std::endl;
    exit(EXIT_FAILURE);
    }
    
    int jnum = numneigh[i];
    int *jlist = firstneigh[i];
    for(int jj = 0; jj < jnum; jj++){
      int j = jlist[jj];
      j &= NEIGHMASK;
      if (type[j] != 1 && type[j] != 2){
       continue;}
      if (type[j] == 3){std::cout<<"type bead in 1 j"<<std::endl;
      exit(EXIT_FAILURE);}
    // 
     // i oz j so indexi od 0 do nlocal, ce je index vecji preskoci, torej ghost indexe preskoci
      double dx = x[i][0] - x[j][0];
      double dy = x[i][1] - x[j][1];
      double dz = x[i][2] - x[j][2];

      double rsq = dx*dx + dy*dy + dz*dz;
      if(rsq <= cutoff*cutoff) {
        neigh_per_atom[i]++;
        nedges++;
      }
    }
  }

  // Cumulative sum of neighbors, for knowing where to fill in the edges tensor
  std::vector<int> cumsum_neigh_per_atom(inum);

  for(int ii = 1; ii < inum; ii++){
    cumsum_neigh_per_atom[ii] = cumsum_neigh_per_atom[ii-1] + neigh_per_atom[ii-1];
  }
  std::cout << "nedges..."<<nedges<<std::endl;
  torch::Tensor pos_tensor = torch::zeros({(ntotal), 3});
  torch::Tensor EF_tensor = torch::zeros({(ntotal), 3});
  torch::Tensor EF1_tensor = torch::zeros({(ntotal), 3});
  torch::Tensor EF2_tensor = torch::zeros({(ntotal), 3});
  torch::Tensor EF3_tensor = torch::zeros({(ntotal), 3});
  torch::Tensor EF4_tensor = torch::zeros({(ntotal), 3});
  torch::Tensor EF5_tensor = torch::zeros({(ntotal), 3});
  torch::Tensor EF6_tensor = torch::zeros({(ntotal), 3});
  torch::Tensor EF7_tensor = torch::zeros({(ntotal), 3});
  torch::Tensor EF8_tensor = torch::zeros({(ntotal), 3});
  torch::Tensor edges_tensor = torch::zeros({2,nedges}, torch::TensorOptions().dtype(torch::kInt64));
  torch::Tensor ij2type_tensor = torch::zeros({(ntotal)}, torch::TensorOptions().dtype(torch::kInt64));

  /*torch::Tensor pos_tensor = torch::zeros({nlocal, 3});
  torch::Tensor EF_tensor = torch::zeros({nlocal, 3});
  torch::Tensor EF1_tensor = torch::zeros({nlocal, 3});
  torch::Tensor EF2_tensor = torch::zeros({nlocal, 3});
  torch::Tensor EF3_tensor = torch::zeros({nlocal, 3});
  torch::Tensor EF4_tensor = torch::zeros({nlocal, 3});
  torch::Tensor EF5_tensor = torch::zeros({nlocal, 3});
  torch::Tensor EF6_tensor = torch::zeros({nlocal, 3});
  torch::Tensor EF7_tensor = torch::zeros({nlocal, 3});
  torch::Tensor EF8_tensor = torch::zeros({nlocal, 3});
  torch::Tensor edges_tensor = torch::zeros({2,nedges}, torch::TensorOptions().dtype(torch::kInt64));
  torch::Tensor ij2type_tensor = torch::zeros({nlocal}, torch::TensorOptions().dtype(torch::kInt64)); */
  
  
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
  auto edges = edges_tensor.accessor<long, 2>();
  auto ij2type = ij2type_tensor.accessor<long, 1>();
  //std::vector<int> tag2i(nlocal);

  // Loop over atoms and neighbors,
  // store edges and _cell_shifts
  // ii follows the order of the neighbor lists,
  // i follows the order of x, f, etc.
  if (debug_mode) printf("Allegro edges: i j rij\n");
#pragma omp parallel for
  for(int ii = 0; ii < ntotal; ii++){
  //for(int ii = 0; ii < nlocal; ii++){
    int i = ilist[ii];
    int itag = tag[i];
    int itype = type[i];
    //std::cout<< " in total loop "<<std::endl;
    if (itype != 1 && itype != 2){
       continue;}
    if (itype == 3) {
    std::cout<<"type bead in 2 i"<<std::endl;
    exit(EXIT_FAILURE);}
    ij2type[i] = type_mapper[itype-1];
   /* if(i < nlocal){
    tag2i[itag-1] = i;}*/
    
    pos[i][0] = x[i][0];
    pos[i][1] = x[i][1];
    pos[i][2] = x[i][2];
    
    EF[i][0] = efield_x[i];
    EF[i][1] = efield_y[i];
    EF[i][2] = efield_z[i];
    
    EF1[i][0] = efield1_x[i];
    EF1[i][1] = efield1_y[i];
    EF1[i][2] = efield1_z[i];
    
    EF2[i][0] = efield2_x[i];
    EF2[i][1] = efield2_y[i];
    EF2[i][2] = efield2_z[i];
    
    EF3[i][0] = efield3_x[i];
    EF3[i][1] = efield3_y[i];
    EF3[i][2] =efield3_z[i];
    
    EF4[i][0] = efield4_x[i];
    EF4[i][1] = efield4_y[i];
    EF4[i][2] = efield4_z[i];
    
    EF5[i][0] = efield5_x[i];
    EF5[i][1] = efield5_y[i];
    EF5[i][2] = efield5_z[i];
    
    EF6[i][0] = efield6_x[i];
    EF6[i][1] = efield6_y[i];
    EF6[i][2] = efield6_z[i];
    
    EF7[i][0] = efield7_x[i];
    EF7[i][1] = efield7_y[i];
    EF7[i][2] = efield7_z[i];
    
    EF8[i][0] = efield8_x[i];
    EF8[i][1] = efield8_y[i];
    EF8[i][2] = efield8_z[i];
    
   
    std::cout << "ii  "<<ii<<" i "<<i<<" itag  "<<itag<<" itype "<<itype<<std::endl;
    //std::cout << "EF[i][0]  "<<EF[i][0]<<" efield_x[itag-1][0] "<<efield_x[itag-1]<<x[i][0]<<"x[j][1]  "<<x[i][1]<<" pos[itag-1][1] "<<pos[i][1]<<"x[i][2]  "<<x[i][2]<<" pos[itag-1][2] "<<pos[i][2]<<std::endl;
    std::cout << " EF[i][0]  "<<EF[i][0]<<" EF[i][1]  "<<EF[i][1]<<" EF[i][2]  "<<EF[i][2]<<" EF1[i][0]  "<<EF1[i][0]<<" EF2[i][0]  "<<EF2[i][0]<<" EF3[i][0]  "<<EF3[i][0]<<" EF4[i][0]  "<<EF4[i][0]<<" EF5[i][0]  "<<EF5[i][0]<<" EF6[i][0]  "<<EF6[i][0]<<" EF7[i][0]  "<<EF7[i][0]<<" EF8[i][0]  "<<EF8[i][0]<<" EF8[i][2]  "<<EF8[i][2]<<std::endl;
   
    if(ii >= inum){continue;}
    
   
    
    
    int jnum = numneigh[i];
    int *jlist = firstneigh[i];

    int edge_counter = cumsum_neigh_per_atom[ii];
    for(int jj = 0; jj < jnum; jj++){
      int j = jlist[jj];
      j &= NEIGHMASK;
      int jtag = tag[j];
      int jtype = type[j];
      if (jtype != 1 && jtype != 2){
       continue;}
      if (jtype == 3) {
      std::cout<<"type bead in 2 j"<<std::endl;
      exit(EXIT_FAILURE);}
      
       
      double dx = x[i][0] - x[j][0];
      double dy = x[i][1] - x[j][1];
      double dz = x[i][2] - x[j][2];

      double rsq = dx*dx + dy*dy + dz*dz;
      if(rsq > cutoff*cutoff) {continue;}
      std::cout << " in j 2 "<<std::endl;
      // TODO: double check order
      edges[0][edge_counter] = i;
      edges[1][edge_counter] = j;

      edge_counter++;

      if (debug_mode) printf("%d %d %.10g\n", itag-1, jtag-1, sqrt(rsq));
    }
  }
  if (debug_mode) printf("end Allegro edges\n");
  std::cout << " writting in input "<<std::endl;
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
  input.insert("atom_types", ij2type_tensor.to(device));
  std::vector<torch::IValue> input_vector(1, input);
  std::cout << " writting in input "<<std::endl;
 
 
  auto output = model.forward(input_vector).toGenericDict();
  
  
  
  std::cout << " writting output "<<std::endl;
  torch::Tensor forces_tensor = output.at("forces").toTensor().cpu();
  std::cout << " writting output forces"<<std::endl;
  auto forces = forces_tensor.accessor<float, 2>();
  std::cout << " writting output forces"<<std::endl;
  //torch::Tensor total_energy_tensor = output.at("total_energy").toTensor().cpu(); WRONG WITH MPI

  torch::Tensor atomic_energy_tensor = output.at("atomic_energy").toTensor().cpu();
  auto atomic_energies = atomic_energy_tensor.accessor<float, 2>();
  float atomic_energy_sum = atomic_energy_tensor.sum().data_ptr<float>()[0];
 std::cout << " writting output energy"<<std::endl;
  //std::cout << "atomic energy sum: " << atomic_energy_sum << std::endl;
  //std::cout << "Total energy: " << total_energy_tensor << "\n";
  //std::cout << "atomic energy shape: " << atomic_energy_tensor.sizes()[0] << "," << atomic_energy_tensor.sizes()[1] << std::endl;
  //std::cout << "atomic energies: " << atomic_energy_tensor << std::endl;

  // Write forces and per-atom energies (0-based tags here)
  eng_vdwl = 0.0;
#pragma omp parallel for reduction(+:eng_vdwl)
  for(int ii = 0; ii < ntotal; ii++){
  //for(int ii = 0; ii < nlocal; ii++){
    int i = ilist[ii];
    int itype = type[i];
    if (itype != 1 && itype != 2){
       continue;}
    if (itype == 3) {
      std::cout<<"type bead in force"<<std::endl;
      exit(EXIT_FAILURE);}
     
      std::cout<<i<<" i "<<itype<<" itype "<<std::endl;
    
    f[i][0] = forces[i][0];
    f[i][1] = forces[i][1];
    f[i][2] = forces[i][2];
    if (eflag_atom && ii < inum) eatom[i] = atomic_energies[i][0];
    if(ii < inum) eng_vdwl += atomic_energies[i][0];
  }std::cout<<" end "<<std::endl;
}
