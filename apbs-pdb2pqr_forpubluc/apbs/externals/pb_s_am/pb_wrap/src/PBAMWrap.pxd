cdef extern from "PBAMStruct.h":

  cdef enum:
    CHR_MAX = 1000
    FIL_MAX = 15
    MOL_MAX = 150
    AT_MAX  = 50000
    XYZRCWIDTH = 5 ## XYZCR = 5 values

##  input
  ctypedef struct PBAMInput:
    double temp_;
    double salt_;
    double idiel_;
    double sdiel_;
    int nmol_;
    char runType_[CHR_MAX];
    char runName_[CHR_MAX];
    int randOrient_;
    double boxLen_;
    int pbcType_;


    ## Electrostatics
    int gridPts_;
    char map3D_[CHR_MAX];

    int grid2Dct_;
    char grid2D_[FIL_MAX][CHR_MAX];
    char grid2Dax_[FIL_MAX][CHR_MAX];
    double grid2Dloc_[FIL_MAX];

    char dxname_[CHR_MAX];

    ## Dynamics
    int ntraj_;
    char termCombine_[CHR_MAX];

    char moveType_[MOL_MAX][CHR_MAX];
    double transDiff_[MOL_MAX];
    double rotDiff_[MOL_MAX];

    int termct_;
    int contct_;

    char termnam_[FIL_MAX][CHR_MAX];
    int termnu_[FIL_MAX][1];
    double termval_[FIL_MAX];
    char confil_[FIL_MAX][CHR_MAX];

    char xyzfil_[MOL_MAX][FIL_MAX][CHR_MAX];
    int xyzct_[MOL_MAX];

##  output
  ctypedef struct PBAMOutput:
    double energies_[MOL_MAX];
    double forces_[MOL_MAX][3];

cdef extern from "PBAMWrap.h":
  PBAMOutput runPBAMSphinxWrap(double xyzrc[][AT_MAX][XYZRCWIDTH],
                               int nmol,
                               int natm[],
                               PBAMInput pbamfin);
