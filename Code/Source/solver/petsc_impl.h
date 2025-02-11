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

    PetscInt  lpPts;    /* number of dofs with lumped parameter BC */
    PetscInt *lpBC_l;   /* O2 index for dofs with lumped parameter BC */
    PetscInt *lpBC_g;   /* PETSc index for dofs with lumped parameter BC */

    PetscInt  lpPts_0;
    PetscInt  lpPts_1;
    PetscInt *lpBC_l_0;   /* O2 index for dofs with lumped parameter BC */
    PetscInt *lpBC_g_0;   /* PETSc index for dofs with lumped parameter BC */
    PetscInt *lpBC_l_1;   /* O2 index for dofs with lumped parameter BC */
    PetscInt *lpBC_g_1;   /* PETSc index for dofs with lumped parameter BC */

    PetscInt  DirPts;   /* number of dofs with Dirichlet BC */
    PetscInt *DirBC;    /* PETSc index for dofs with Dirichlet BC */

    PetscInt  DirPts_0;   /* number of dofs with Dirichlet BC */
    PetscInt *DirBC_0;    /* PETSc index for dofs with Dirichlet BC */
    PetscReal *svFSI_DirBC_0;    /* svFSI Dirichlet BC */
    PetscInt  DirPts_1;   /* number of dofs with Dirichlet BC */
    PetscInt *DirBC_1;    /* PETSc index for dofs with Dirichlet BC */
    PetscReal *svFSI_DirBC_1;    /* svFSI Dirichlet BC */

    Vec       b;        /* rhs/solution vector of owned vertices */
    Vec       b_n[2];   /* block vectors */
    Mat       A;        /* stiffness matrix */
    Mat       A_mn[4];  /* block matrices */
    KSP       ksp;      /* linear solver context */

    PetscBool bnpc;     /* whether block nested preconditioner is activated */
    PetscBool rcs;      /* whether rcs preconditioner is activated */
    Vec       Dr;       /* diagonal matrix from row maxabs */
    Vec       Dc;       /* diagonal matrix from col maxabs */
} LSCtx;

void petsc_initialize(const PetscInt nNo, const PetscInt mynNo, 
    const PetscInt nnz, const PetscInt nEq, const PetscInt *svFSI_ltg, 
    const PetscInt *svFSI_map, const PetscInt *svFSI_rowPtr, 
    const PetscInt *svFSI_colPtr, char *inp);

void petsc_create_linearsystem(const PetscInt dof, const PetscInt iEq, const PetscInt nEq, 
    const PetscReal *svFSI_DirBC, const PetscReal *svFSI_lpBC);

void petsc_create_linearsolver(const consts::PreconditionerType lsType, const consts::PreconditionerType pcType, 
    const PetscInt kSpace, const PetscInt maxIter, const PetscReal relTol, 
    const PetscReal absTol, const consts::EquationType phys, const PetscInt dof, 
    const PetscInt nnz, const PetscInt iEq, const PetscInt nEq);

void petsc_set_values(const PetscInt dof, const PetscInt iEq, const PetscReal *R, 
    const PetscReal *Val, const PetscReal *svFSI_DirBC, const PetscReal *svFSI_lpBC);

PetscErrorCode petsc_solve(PetscReal *resNorm,  PetscReal *initNorm,  PetscReal *dB, 
    PetscReal *execTime, bool *converged, PetscInt *numIter, 
    PetscReal *R, const PetscInt maxIter, const PetscInt dof, 
    const PetscInt iEq);

void petsc_destroy_all(const PetscInt);

PetscErrorCode petsc_create_lhs(const PetscInt, const PetscInt, const PetscInt,  
                                const PetscInt *, const PetscInt *, 
                                const PetscInt *, const PetscInt *);

PetscErrorCode petsc_create_bc(const PetscInt, const PetscInt, const PetscReal *, 
                               const PetscReal *);

PetscErrorCode petsc_create_splitbc(const PetscInt, const PetscInt, const PetscReal *, 
                               const PetscReal *);

PetscErrorCode petsc_create_vecmat(const PetscInt, const PetscInt, const PetscInt);

PetscErrorCode petsc_create_splitvecmat(const PetscInt, const PetscInt, const PetscInt);

PetscErrorCode petsc_set_vec(const PetscInt, const PetscInt, const PetscReal *);

PetscErrorCode petsc_set_splitvec(const PetscInt, const PetscInt, const PetscReal *);

PetscErrorCode petsc_set_mat(const PetscInt, const PetscInt, const PetscReal *);

PetscErrorCode petsc_set_splitmat(const PetscInt, const PetscInt, const PetscReal *);

PetscErrorCode petsc_set_bc(const PetscInt, const PetscReal *, const PetscReal *);

PetscErrorCode petsc_set_splitbc(const PetscInt, const PetscReal *, const PetscReal *);

PetscErrorCode petsc_set_pcfieldsplit(const PetscInt, const PetscInt);

PetscErrorCode petsc_pc_rcs(const PetscInt, const PetscInt);


PetscErrorCode petsc_debug_save_vec(const char *, Vec);
PetscErrorCode petsc_debug_save_mat(const char *, Mat);

//---------------------------------------------------------------
// BLOCK NESTED PRECONDITIONER 
// --------------------------------------------------------------
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

// forward declaration
class BlockNestedPC_InternalLinearSolver;

typedef struct 
{    
    Mat A00, A01, A10, A11;                        // Matrices involved in the operation
    Vec v_0, v_1, v_1_tmp;                         // Temporary vectors
    BlockNestedPC_InternalLinearSolver *ASolver;   // Solver object for internal operations
} ShellMatrixContext;

class BlockNestedPreconditioner
{
  public:
    // --------------------------------------------------------------
    // Input the relative tolerance for the _0 solver and the _1 solver
    // _0 is associated with the A_00 matrix
    // _1 is associated with the Schur complement (matrix-free algorithm).
    // --------------------------------------------------------------
    BlockNestedPreconditioner( const PetscReal rtol0, 
        const PetscReal atol0, const PetscReal dtol0, const PetscInt maxit0,
        const PetscReal rtol1, const PetscReal atol1, const PetscReal dtol1, 
        const PetscInt maxit1);

    // Destructor
    ~BlockNestedPreconditioner();

    // Initialize the index sets
    /* 
    PetscErrorCode Initialize( const PetscInt dof, const PetscInt nnz);
    */
    // Application function for the block nested preconditioner to be used 
    // by the FGMRES solver
    PetscErrorCode BlockNestedPC_Apply(PC pc, Vec r, Vec z);

    // Set the tangent matrix for the block nested preconditioner and 
    // create the block matrices
    PetscErrorCode BlockNestedPC_SetMatrix( Mat A );

    PetscErrorCode SetApproximateSchur(Mat **A);

  private:
  
    //Mat tangent_matrix;                  /* Full tangent matrix */
    Mat **A_mn;                            /* Block matrices */
    Vec *r_m;
    Vec *z_m;
    Mat Ps;                               /* Approximate Schur complement matrix */
    //Vec z_0, z_1;                       /* Block Solution/temporary vectors */
    //IS row_velocity_is, row_pressure_is;  /* Index sets for row splitting */
    //IS col_velocity_is, col_pressure_is;  /* Index sets for column splitting */
    PetscInt local_size_v, local_size_p ; /* Size of the index sets */

    BlockNestedPC_InternalLinearSolver * solver_0; /* matrix-free solver for Schur complement */

    BlockNestedPC_InternalLinearSolver * solver_1; /* explicit solver for A00 */
};

class BlockNestedPC_InternalLinearSolver 
{
  public:
    KSP ksp;

    // Default KSP object: rtol = 1.0e-5, atol = 1.0e-50, dtol = 1.0e50
    // maxits = 10000, and the user may reset the parameters by options
    BlockNestedPC_InternalLinearSolver();
    
    // Construct KSP with input tolerances and maximum iteration
    BlockNestedPC_InternalLinearSolver( const double &input_rtol, const double &input_atol, 
        const double &input_dtol, const int &input_maxits );
    
    // Construct KSP with input tolerances and maximum iteration with prefix
    BlockNestedPC_InternalLinearSolver( const double &in_rtol, const double &in_atol,
        const double &in_dtol, const int &in_maxits,
        const char * const &ksp_prefix, const char * const &pc_prefix);

    // Destructor 
    ~BlockNestedPC_InternalLinearSolver();

    // Assign a matrix K to the linear solver object and a matrix P for 
    // the preconditioner if different from K
    PetscErrorCode SetOperator(const Mat &A) {PetscCall(KSPSetOperators(ksp, A, A));}

    PetscErrorCode SetOperator(const Mat &A, const Mat &P) {PetscCall(KSPSetOperators(ksp, A, P));}

    // Solve a linear problem A x = b
    // with x plain vector, with no ghost entries.
    PetscErrorCode Solve( const Vec &G, Vec &out_sol, const bool &isPrint=true );

    PetscErrorCode Solve( const Mat &K, const Vec &G, Vec &out_sol, 
        const bool &isPrint=true );

    // Link the solver to a preconditioner context
    PetscErrorCode GetPC( PC *prec ) const {PetscCall(KSPGetPC(ksp, prec));}

    // Set the Schur Complement object
    //void SetSchurComplement(MatrixFreeSchurComplement *Schur) { SchurComplement = Schur; }
    
    // Get matrix 
    //Mat GetSchur() const {return SchurComplement->GetMatrix();}

    // Get the iteration number
    int get_ksp_it_num() const
    {int it_num; KSPGetIterationNumber(ksp, &it_num); return it_num;}

    // Get maximum iteration number for this linear solver
    int get_ksp_maxits() const
    { 
      int mits; 
      KSPGetTolerances(ksp, PETSC_NULL, PETSC_NULL, PETSC_NULL, &mits);
      return mits;
    }

    // Print the ksp info on screen
    void print_info() const
    {
      PetscPrintf(MPI_COMM_WORLD, " --------------------------------------\n");
      KSPView(ksp, PETSC_VIEWER_STDOUT_WORLD);
      PetscPrintf(MPI_COMM_WORLD, " --------------------------------------\n");  
    }

    //  Monitor the Krylov subspace method behavior
    void Monitor() const;

  private: 
    // Application of Schur complement to a vector for matrix-free algorithm
    // MatrixFreeSchurComplement * SchurComplement;

    // relative, absolute, divergence tolerance
    const PetscReal rtol, atol, dtol;
    
    // maximum number of iterations 
    const PetscInt maxits;
};

class MatrixFreeSchurComplement 
{
  public:
    // Constructor: initialized the matshell context 
    MatrixFreeSchurComplement(Mat **K, BlockNestedPC_InternalLinearSolver* solver_0);

    // Destructor
    ~MatrixFreeSchurComplement();

    // Get Schur matrix
    Mat GetSchurMatrix() const {return S;}

    // Static method to handle the Schur complement multiplication
    static PetscErrorCode SchurComplMult(Mat mat, Vec x, Vec y);
    
    PetscErrorCode ApplySchurComplement();

  private:
    Mat S;
    // Method defining the Schur complement matrix-free procedure
    //PetscErrorCode SchurComplMult(Mat mat, Vec x, Vec y);

    // Context for the shell matrix
    ShellMatrixContext *MatShellCtx; 
};

PetscErrorCode MatCreatePreallocator(PetscInt m, PetscInt n, Mat *A);

PetscErrorCode MatCreatePreallocSubMat(PetscInt m, PetscInt n, Mat *B, Mat *A);

#endif
