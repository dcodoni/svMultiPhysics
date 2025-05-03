/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 *
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files (the
 * "Software"), to deal in the Software without restriction, including
 * without limitation the rights to use, copy, modify, merge, publish,
 * distribute, sublicense, and/or sell copies of the Software, and to
 * permit persons to whom the Software is furnished to do so, subject
 * to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included
 * in all copies or substantial portions of the Software.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
 * IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
 * TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
 * PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER
 * OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
 * LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 * NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

// This file contains PETSc-dependent data structures. 

#ifndef PETSC_INTERFACE_H
#define PETSC_INTERFACE_H

#include <petscksp.h>
#include <petscao.h>
#include <unistd.h>
#include <stdbool.h>

#include "consts.h"

//--------
// LHSCtx
//---------
// PETSc lhs context. 
//
typedef struct {
    PetscBool created;  /* Whether petsc lhs is created */

    PetscInt  nNo;      /* local number of vertices */
    PetscInt  mynNo;    /* number of owned vertices */

    PetscInt *map;      /* local to local mapping, map[O2] = O1 */
    PetscInt *ltg;      /* local to global in PETSc ordering */
    PetscInt *ghostltg; /* local to global in PETSc ordering */
    PetscInt *ghostltg_0;
    PetscInt *ghostltg_1;
    PetscInt *rowPtr;   /* row pointer for adjacency info */
    PetscInt *colPtr;   /* column pointer for adjacency info */
} LHSCtx;

//-------
// LSCtx
//-------
// PETSc linear solver context. 
//
typedef struct {
    PetscBool created;  /* Whether mat and vec is created */
    const char *pre;      /* option prefix for different equations */

    PetscInt nResFaces; /* number of faces where Resistance BC is applied */
    PetscInt *lpPts;    /* number of dofs with lumped parameter BC */
    PetscInt **lpBC_l;   /* O2 index for dofs with lumped parameter BC */
    PetscInt **lpBC_g;   /* PETSc index for dofs with lumped parameter BC */

    /*
      Global arrays for the lumped parameter BC
    */
    PetscInt *lpPts_all;   /* same as lpPts but gathering data from all tasks */
    PetscInt **lpBC_l_all; /* same as lpBC_l but gathering data from all tasks */
    PetscInt **lpBC_g_all; /* same as lpBC_g but gathering data from all tasks */
    PetscReal **lpBC_Val_all; /* svFSI lumped parameter BC per face from all tasks */

    PetscInt  DirPts;   /* number of dofs with Dirichlet BC */
    PetscInt *DirBC;    /* PETSc index for dofs with Dirichlet BC */

    /* 
      _0 refer to velocity dofs when block-iterative procedure is used 
      _1 refer to pressure dof when block-iterative procedure is used 
    */
    PetscInt *lpPts_0;
    PetscInt **lpBC_l_0;
    PetscInt **lpBC_g_0;
    PetscReal **svFSI_lpBC_0; /* svFSI lumped parameter BC for the velocity dofs */
    PetscInt *lpPts_0_all;   
    PetscInt **lpBC_l_0_all; 
    PetscInt **lpBC_g_0_all; 
    PetscReal **lpBC_Val_0_all;
    PetscInt *lpPts_1;
    PetscInt **lpBC_l_1;
    PetscInt **lpBC_g_1;
    PetscReal **svFSI_lpBC_1; /* svFSI lumped parameter BC for the pressure dof */
    PetscInt *lpPts_1_all;   
    PetscInt **lpBC_l_1_all; 
    PetscInt **lpBC_g_1_all; 
    PetscReal **lpBC_Val_1_all;

    PetscInt  DirPts_0;   
    PetscInt *DirBC_0;    
    PetscReal *svFSI_DirBC_0; /* svFSI Dirichlet BC for the velocity dofs */
    PetscInt  DirPts_1;   
    PetscInt *DirBC_1;    
    PetscReal *svFSI_DirBC_1; /* svFSI Dirichlet BC for the pressure dof */

    Vec       b;        /* rhs/solution vector of owned vertices */
    Vec       b_n[2];   /* rhs/solution subvectors of owned vertices */
    Mat       A;        /* stiffness matrix */
    Mat       A_mn[4];  /* stiffness submatrices */
    KSP       ksp;      /* linear solver context */

    PetscBool block_iterative_pc; /* whether a block-iterative preconditioner is used */
    PetscBool rcs;      /* whether rcs preconditioner is activated */
    Vec       Dr;       /* diagonal matrix from row maxabs */
    Vec       Dc;       /* diagonal matrix from col maxabs */
} LSCtx;

void petsc_initialize(const PetscInt nNo, const PetscInt mynNo, 
    const PetscInt nnz, const PetscInt nEq, const PetscInt *svFSI_ltg, 
    const PetscInt *svFSI_map, const PetscInt *svFSI_rowPtr, 
    const PetscInt *svFSI_colPtr, char *inp);

// void petsc_create_linearsystem(const PetscInt dof, const PetscInt iEq, const PetscInt nEq, 
//     const PetscReal *svFSI_DirBC, const PetscReal *svFSI_lpBC);
void petsc_create_linearsystem(const PetscInt dof, const PetscInt iEq, const PetscInt nEq, const PetscInt nTask,
     const PetscReal *svFSI_DirBC, PetscReal** svFSI_lpBC, const PetscInt nFacesRes);

void petsc_create_linearsolver(const consts::SolverType lsType, const consts::PreconditionerType pcType, 
    const BlockIterativePCparams pc_params, const PetscInt kSpace, const PetscInt maxIter, const PetscReal relTol, 
    const PetscReal absTol, const consts::EquationType phys, const PetscInt dof, 
    const PetscInt nnz, const PetscInt iEq, const PetscInt nEq);

void petsc_set_values(const PetscInt dof, const PetscInt iEq, const PetscReal *R, 
    const PetscReal *Val, const PetscReal *svFSI_DirBC, PetscReal** svFSI_lpBC);

PetscErrorCode petsc_solve(PetscReal *resNorm,  PetscReal *initNorm,  PetscReal *dB, 
    PetscReal *execTime, bool *converged, PetscInt *numIter, 
    PetscReal *R, const PetscInt maxIter, const PetscInt dof, 
    const PetscInt iEq);

void petsc_destroy_all(const PetscInt);

PetscErrorCode petsc_create_lhs(const PetscInt, const PetscInt, const PetscInt,  
                                const PetscInt *, const PetscInt *, 
                                const PetscInt *, const PetscInt *);

PetscErrorCode petsc_create_bc(const PetscInt, const PetscInt, const PetscInt,
                               const PetscReal *, PetscReal**);

PetscErrorCode petsc_create_splitbc(const PetscInt, const PetscInt, const PetscInt,
                                    const PetscReal *, PetscReal **);

PetscErrorCode petsc_create_vecmat(const PetscInt, const PetscInt, const PetscInt);

PetscErrorCode petsc_create_splitvecmat(const PetscInt, const PetscInt, const PetscInt);

PetscErrorCode petsc_set_vec(const PetscInt, const PetscInt, const PetscReal *);

PetscErrorCode petsc_set_splitvec(const PetscInt, const PetscInt, const PetscReal *);

PetscErrorCode petsc_set_mat(const PetscInt, const PetscInt, const PetscReal *);

PetscErrorCode petsc_set_splitmat(const PetscInt, const PetscInt, const PetscReal *);

PetscErrorCode petsc_set_bc(const PetscInt, const PetscReal *, PetscReal**);

PetscErrorCode petsc_set_splitbc(const PetscInt);

PetscErrorCode petsc_set_pcfieldsplit(const PetscInt, const PetscInt);

PetscErrorCode petsc_pc_rcs(const PetscInt, const PetscInt);


PetscErrorCode petsc_debug_save_vec(const char *, Vec);
PetscErrorCode petsc_debug_save_mat(const char *, Mat);

//----------
// PC_LSCtx
//----------
/* 
  General class for the linear solvers used internally by custom 
  block-iterative PC:
  - creates a ksp context, sets the tolerances, create ksp prefix
    for options
  - set operators for the solver: LHS matrix (A) and PC matrix (P)
  - solve the linear system
  - get the preconditioner context
  - debugging and info tools for linear solver
*/
class PC_LSCtx
{
  public:
    KSP ksp;

    PC_LSCtx( const double &in_rtol, const double &in_atol,
        const double &in_dtol, const int &in_maxits,
        const char * const &ksp_prefix, const char * const &pc_prefix);

    ~PC_LSCtx();

    PetscErrorCode SetOperator(const Mat &A) {PetscCall(KSPSetOperators(ksp, A, A));}
    PetscErrorCode SetOperator(const Mat &A, const Mat &P) {PetscCall(KSPSetOperators(ksp, A, P));}

    PetscErrorCode Solve( const Vec &in_R, Vec &out_Sol, const bool &isPrint=true );

    PetscErrorCode GetPC( PC *prec ) const {PetscCall(KSPGetPC(ksp, prec));}

    /* 
      Debugging and Info tools 
    */
    int get_ksp_it_num() const
    {int it_num; KSPGetIterationNumber(ksp, &it_num); return it_num;}

    int get_ksp_maxits() const
    { 
      int mits; 
      KSPGetTolerances(ksp, PETSC_NULL, PETSC_NULL, PETSC_NULL, &mits);
      return mits;
    }

    void print_info() const
    {
      PetscPrintf(MPI_COMM_WORLD, " --------------------------------------\n");
      KSPView(ksp, PETSC_VIEWER_STDOUT_WORLD);
      PetscPrintf(MPI_COMM_WORLD, " --------------------------------------\n");  
    }

    void Monitor() const;

    void get_ksp_estimate_condition() const;

  private: 
    const PetscReal rtol, atol, dtol;
    const PetscInt maxits;
};

//------------------
// BlockIterative_PC
//------------------
/*
  Block-iterative preconditioners are based on the following  
  factorization of the matrix A (LHS of the system):
  L * D * U = A = | A, B |
                  | C, D |

  | I,        0 | * | A, 0 | * | I, A^{-1}*B | = A
  | C*A^{-1}, I |   | 0, S |   | 0, I        |
*/

/* 
  Abstract base class for block-iterative preconditioners, ensuring that 
  all derived classes implement the necessary methods
*/
class BlockIterative_Preconditioner 
{
  private:
      Mat A;                   /* System matrix (LHS) defined as MATNEST */
      Mat **subA;              /* Submatrices of the system matrix */
      PetscInt size_0, size_1; /* Local sizes of _0 and _1 systems */
      PC pc;                   /* Blokc-iterative preconditioner object */
  
  public:
    BlockIterative_Preconditioner(PC pc_) : pc(pc_), A(NULL) {}
    virtual ~BlockIterative_Preconditioner() {}
  
    /* 
      Set the type of PC, attach a context to PC and set the 
      application function which defines the PC itself
    */
    void CreatePC() {
      PCSetType(pc, PCSHELL);
      PCShellSetContext(pc, this);
      PCShellSetApply(pc, applyPC);
    }
    /*
      Initialize the internal solvers for the block-iterative PC
    */
    void initialize_solvers(const PetscReal rtol0, const PetscReal atol0, 
      const PetscReal dtol0, const PetscInt maxit0, std::string prec_0,
      const PetscReal rtol1, const PetscReal atol1, const PetscReal dtol1, 
      const PetscInt maxit1, std::string prec_1)
    {
      initialize(rtol0, atol0, dtol0, maxit0, prec_0, 
        rtol1, atol1, dtol1, maxit1, prec_1);
    }
    /*
      Set the matrix for the block-iterative PC.
      Anything specific to a particular PC (SCR, SIMPLE, ...) is
      also set here.
    */
    void setMatrix(Mat A_) 
    {
      A = A_;
      MatNestGetSubMats(A, NULL, NULL, &subA);
      MatGetLocalSize(subA[0][1], &size_0, &size_1);
      setup();
    }
    
    Mat getMatrix() { return A; }

    Mat** getSubMatrix() { return subA; }

    PetscInt GetSize(PetscInt s_) 
    { 
      if (s_ == 0) return size_0;
      else return size_1;
    }

    /*
      The function applyPC is a static member function that calls
      another function apply() which is defined differently for each
      type of PC
    */
    static PetscErrorCode applyPC(PC pc, Vec x, Vec y) {
      void *ctx;
      PCShellGetContext(pc, &ctx);
      auto *blockPC = static_cast<BlockIterative_Preconditioner *>(ctx);
      return blockPC->apply(pc, x, y);
    }

    /*
        Set the type of the preconditioner for the internal solvers
      */
     PetscErrorCode set_internal_pc_type(PC_LSCtx *internal_solver, PC pc_internal, std::string pc_type);

    /*
      Virtual function that must be implemented by derived classes
      and they are specific for each PC type
    */
    virtual void initialize(const PetscReal rtol0, const PetscReal atol0, 
      const PetscReal dtol0, const PetscInt maxit0, std::string prec_0,
      const PetscReal rtol1, const PetscReal atol1, const PetscReal dtol1, 
      const PetscInt maxit1, std::string prec_1) = 0;

    virtual void setup() = 0;

    virtual PetscErrorCode apply(PC pc, Vec x, Vec y) = 0;
};

//-------------------
// SCR_Preconditioner
//-------------------

/* Iterative solution method with the nested block preconditioning
Linear system is: A * x = r with the following block structure:
A = [ A_00, A_01   [ xu    = [ r_0
      A_10, A_11 ]   xp ]      r_1 ]
The method involves the FGMRES with a varying preconditioner PC
for each iteration. The procedure involves three steps:
1.  Outer solver: FGMRES that generates a Krylov subspace by applying
    A * PC^{-1} to a vector.
2.  Intermediate solver: Gives the application of the PC^{-1} to a
    vector.
3.  Inner solver: Matrix-free algorithm for the application of the
    Schur complement to a vector.
DOI: https://doi.org/10.1016/j.cma.2020.113122 
*/
// --------------------------------------------------------------
/* Notes:
Intermediate solver:
S xp = r_1 - A_10 inv(A_00) r_0
A_00 xu = r_0 - A_01 xp
wherein S := A_11 - A_10 inv(A_00) A_01 is the Schur complement,
not explicitly formed but solved by in the inner solver with a 
a matrix-free algorithm. 
*/
// --------------------------------------------------------------

/* 
  Derived class for the Schur Complement Reduction (SCR) preconditioner
*/
class SCR_Preconditioner : public BlockIterative_Preconditioner 
{
  private:
      Vec *subR;          /* subvectors for the residual */     
      Vec *subZ;          /* subvectors for the Krylov member of FGMRES */
      Mat Ps;             /* Approximate Schur complement matrix */
      Mat S;              /* Schur complement matrix (matrix-free procedure)*/
      PC_LSCtx *solver_0; /* linear solver for A_00 submatrix */
      PC_LSCtx *solver_1; /* linear solver for Schur complement */
      std::string pc_type_0; /* type of preconditioner for solver_0 */
      std::string pc_type_1; /* type of preconditioner for solver_1 */

  public:
      SCR_Preconditioner(PC pc) : BlockIterative_Preconditioner(pc) {}
      
      ~SCR_Preconditioner() override;

      void initialize_internal_solver(const PetscReal rtol0, const PetscReal atol0, 
        const PetscReal dtol0, const PetscInt maxit0, std::string prec_0,
        const PetscReal rtol1, const PetscReal atol1, const PetscReal dtol1, 
        const PetscInt maxit1, std::string prec_1)
      {
        solver_0 = new PC_LSCtx(rtol0, atol0, dtol0, maxit0, "ls0_", "pc0_");
        solver_1 = new PC_LSCtx(rtol1, atol1, dtol1, maxit1, "ls1_", "pc1_");

        pc_type_0 = prec_0;
        pc_type_1 = prec_1;
      }
      
      /* 
        Implements the logic of the SCR preconditioner 
      */
      PetscErrorCode SCR_PCApply(PC pc, Vec x, Vec y);
    
      /* 
        Compute the approximate Schur complement using the diag(A_00) 
         and saves it in Ps
      */
      PetscErrorCode SetApproximateSchur();

      /*
        Implements the application of the Schur complement to a vector
      */
      static PetscErrorCode SCR_SchurApply(Mat mat, Vec x, Vec y);

      void initialize(const PetscReal rtol0, const PetscReal atol0, 
        const PetscReal dtol0, const PetscInt maxit0, std::string prec_0,
        const PetscReal rtol1, const PetscReal atol1, const PetscReal dtol1, 
        const PetscInt maxit1, std::string prec_1) override 
      {
        this->initialize_internal_solver(rtol0, atol0, dtol0, maxit0, prec_0, 
          rtol1, atol1, dtol1, maxit1, prec_1);
      }

      PetscErrorCode apply(PC pc, Vec x, Vec y) override 
      {
        this->SCR_PCApply(pc, x, y);
      }

      void setup() override 
      {
        MatCreateShell(MPI_COMM_WORLD, GetSize(1) , GetSize(1), PETSC_DECIDE, PETSC_DECIDE, this, &S);
        MatShellSetOperation(S, MATOP_MULT, (void (*)(void))SCR_SchurApply);
        this->SetApproximateSchur();
      }
};

//----------------------
// SIMPLE_Preconditioner
//----------------------
/* Iterative solution method with the nested block preconditioning
Linear system is: A * x = r with the following block structure:
A = [ A_00, A_01   [ xu    = [ r_0
      A_10, A_11 ]   xp ]      r_1 ]
The method involves the FGMRES with a varying preconditioner PC
for each iteration. The procedure involves three steps:
1.  Outer solver: FGMRES that generates a Krylov subspace by applying
    A * PC^{-1} to a vector.
2.  Intermediate solver: Gives the application of the PC^{-1} to a
    vector.
3.  Inner solver: the Schur is approximated by using the inverse of 
    the diagonal of A_00 and solved with GMRES.
DOI: https://doi.org/10.1016/j.cma.2020.113122 
*/
// --------------------------------------------------------------
/* Notes:
Intermediate solver:
Ps xp = r_1 - A_10 inv(A_00) r_0
A_00 xu = r_0 - A_00 (diag(A_00))^{-1} A_01 xp
wherein Ps := A_11 - A_10 (diag(A_00))^{-1} A_01 is the approximation 
of the Schur complement.
*/
// --------------------------------------------------------------

/* 
  Derived class for the SIMPLE preconditioner
*/
class SIMPLE_Preconditioner : public BlockIterative_Preconditioner 
{
  private:
      Vec *subR;          /* subvectors for the residual */     
      Vec *subZ;          /* subvectors for the Krylov member of FGMRES */
      Mat Ps;             /* Approximate Schur complement matrix */
      PC_LSCtx *solver_0; /* linear solver for A_00 submatrix */
      PC_LSCtx *solver_1; /* linear solver for Schur complement */
      std::string pc_type_0; /* type of preconditioner for solver_0 */
      std::string pc_type_1; /* type of preconditioner for solver_1 */

  public:
      SIMPLE_Preconditioner(PC pc) : BlockIterative_Preconditioner(pc) {}
      
      ~SIMPLE_Preconditioner() override;

      void initialize_internal_solver(const PetscReal rtol0, const PetscReal atol0, 
        const PetscReal dtol0, const PetscInt maxit0, std::string prec_0,
        const PetscReal rtol1, const PetscReal atol1, const PetscReal dtol1, 
        const PetscInt maxit1, std::string prec_1)
      {
        solver_0 = new PC_LSCtx(rtol0, atol0, dtol0, maxit0, "ls0_", "pc0_");
        solver_1 = new PC_LSCtx(rtol1, atol1, dtol1, maxit1, "ls1_", "pc1_");

        pc_type_0 = prec_0;
        pc_type_1 = prec_1;
      }

      /* 
        Implements the logic of the SIMPLE preconditioner 
      */
      PetscErrorCode SIMPLE_PCApply(PC pc, Vec x, Vec y);

      /* 
        Compute the approximate Schur complement using the diag(A_00) 
         and saves it in Ps
      */
      PetscErrorCode SetApproximateSchur();

      void initialize(const PetscReal rtol0, const PetscReal atol0, 
      const PetscReal dtol0, const PetscInt maxit0, std::string prec_0,
      const PetscReal rtol1, const PetscReal atol1, const PetscReal dtol1, 
      const PetscInt maxit1, std::string prec_1) override 
      {
        this->initialize_internal_solver(rtol0, atol0, dtol0, maxit0, prec_0, 
          rtol1, atol1, dtol1, maxit1, prec_1);
      }

      PetscErrorCode apply(PC pc, Vec x, Vec y) override 
      {
        this->SIMPLE_PCApply(pc, x, y);
      }

      void setup() override 
      {
        this->SetApproximateSchur();
      }
};

//-------------------------
// MAP: pc type to function
//-------------------------
/*
  Factory mapping block-iterative preconditioner type to constructor functions
*/
std::unordered_map<std::string, std::function<BlockIterative_Preconditioner*(PC)>> block_iterative_pc_map = {
  {"petsc-scr",    [](PC pc) { return new SCR_Preconditioner(pc); }},
  {"petsc-simple", [](PC pc) { return new SIMPLE_Preconditioner(pc); }}
};

PetscErrorCode MatCreatePreallocator(PetscInt m, PetscInt n, Mat *A);

PetscErrorCode MatCreatePreallocSubMat(PetscInt m, PetscInt n, Mat *B, Mat *A);

#endif
