# Training with masking out some force labels.
#
# !! This is in BETA !!
# !! please check your results carefully and report any issues you find!
#
# NequIP supports masking out ground truth label values from loss functions and metrics 
# by setting them to NaN. All statistics will be correctly computed with denominators
# taken from the number of un-masked labels.

# !! PLEASE NOTE: `minimal_mask.yaml` is meant as a _minimal_ example of a tiny, fast
#                 training that shows how to mask out some force labels.
#                 These are NOT recommended hyperparameters for real applications!
#                 Please see `example.yaml` for a reasonable starting point.

# general
root: ./results/train_1
run_name: CG_dna
seed: 123456
dataset_seed: 123456
append: true
default_dtype: float32


model_builders:
 - allegro.model.Allegro
 # the typical model builders from `nequip` can still be used:
 - PerSpeciesRescale
 - ForceOutput
   #- RescaleEnergyEtc

# network --- see `configs/minimal.yaml`
r_max: 12.0
avg_num_neighbors: auto

num_basis: 12
BesselBasis_trainable: true
PolynomialCutoff_p: 24

l_max: 2
parity: o3_restricted

num_layers: 1
env_embed_multiplicity: 16 #amaury veliko do 64
embed_initial_edge: true

two_body_latent_mlp_latent_dimensions: [32,64] #mogoce tukaj vec zaradi efielda
two_body_latent_mlp_nonlinearity: silu
two_body_latent_mlp_initialization: uniform

latent_mlp_latent_dimensions: [64,64,64]
latent_mlp_nonlinearity: silu
latent_mlp_initialization: uniform
latent_resnet: true

env_embed_mlp_latent_dimensions: []
env_embed_mlp_nonlinearity: null
env_embed_mlp_initialization: uniform

# - end allegro layers -

# Final MLP to go from Allegro latent space to edge energies:
edge_eng_mlp_latent_dimensions: [32]
edge_eng_mlp_nonlinearity: null
edge_eng_mlp_initialization: uniform

# data set
dataset: ase
# If you don't have this file, please run `generate_slab.py` in this directory to generate the toy data:
dataset_file_name: ./input_data/config_for_train.xyz
#dataset_key_mapping:
#   tags: tags
key_mapping:
   Z: atomic_numbers # atomic species, integers
   energy: total_energy # total potential eneriges to train to
   force: forces # atomic forces to train to
   pos: pos
   pbc: pbc

ase_args:
   format: extxyz

dataset_include_keys: ["tags"]

# ! This important line adds a "pre-transform" to the dataset that processes the `AtomicData` *after* it is loaded and processed
#   from the original data file, but before it is cached to disk for later training use.  This function can be anything, but here
#   we use `nequip.data.transforms.MaskByAtomTag` to make ground truth force labels on atoms with a certain tag NaN (i.e. masked)
#   We can mask multiple tags if we like, or mask other per-atom fields.  In this case, we mask out tag 0, which is subsurface
#   atoms in the toy data from `generate_slab.py`.
#dataset_pre_transform: !!python/object:nequip.data.transforms.MaskByAtomTag {'tag_values_to_mask': [0], 'fields_to_mask': ['forces']}

chemical_symbol_to_type:
   Na: 0
   Cl: 1
   P: 2
   C: 3
   O: 4
   N: 5

#include_keys: ["tags"]

dataset_pre_transform: !!python/object:nequip.data.transforms.MaskByAtomTag {'tag_values_to_mask': [0], 'fields_to_mask': ['forces']}

wandb: false
    #wandb_project: allegro-tutorial
verbose: debug
    #log_batch_freq: 10

# training
n_train: 160000
n_val: 40000
batch_size: 16
max_epochs: 400
learning_rate: 0.002 #tudi 0.001
train_val_split: sequential
shuffle: false
#report_init_validation: true
metrics_key: validation_loss_f


# use an exponential moving average of the weights
use_ema: true
ema_decay: 0.99
ema_use_num_updates: true


# loss function
loss_coeffs:
  forces:
   - 1
   #- PerSpeciesMSELoss
   - MSELoss
   #- MSELoss
     # ! We have to specify that the force loss function should treat NaNs as masked data, rather than propagating them as usual
     #   (If we forget this, everything will become a NaN.)
   - {"ignore_nan": true}
     #total_energy:
     #- 1
     #- PerAtomMSELoss

metrics_components:
   #- - forces
   #  - mae
      # ! We have to specify that the force metrics should treat NaNs as masked data, rather than propagating them as usual
      #   (If we forget this, we will just get 0 due to the NaNs.)
   #  - ignore_nan: True
   - - forces
     - rmse
      # same as above
     - ignore_nan: True  
     #- PerSpecies: True
     #  report_per_component: False
   - - forces
     - mae 
     - ignore_nan: True
       #- - total_energy
       #- mae

optimizer_name: Adam
#optimizer_params:
optimizer_amsgrad: false
optimizer_betas: !!python/tuple
 - 0.9
 - 0.999
optimizer_eps: 1.0e-08
optimizer_weight_decay: 0.

lr_scheduler_name: ReduceLROnPlateau
lr_scheduler_patience: 50
lr_scheduler_factor: 0.5

early_stopping_upper_bounds:
  cumulative_wall: 604800.

early_stopping_lower_bounds:
  LR: 1.0e-5

early_stopping_patiences:
  validation_loss: 100

