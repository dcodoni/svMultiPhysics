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

#ifndef TRILINOS_LINEAR_SOLVER_H
#define TRILINOS_LINEAR_SOLVER_H
/*!
  \file    trilinos_linear_solver.h
  \brief   wrap Trilinos solver functions
*/

/**************************************************************/
/*                          Includes                          */
/**************************************************************/

#include <stdio.h>
#include <vector>
#include <iostream>
#include "mpi.h"
#include <time.h>
#include <numeric>

// Theuchos includes
#include "Teuchos_RCP.hpp"
#include "Teuchos_DefaultComm.hpp"

// Epetra includes
#include "Epetra_MpiComm.h" //include MPI communication
#include "Epetra_Map.h" //need to create block map
#include "Epetra_FEVbrMatrix.h" //sparse matrix for FE
#include "Epetra_FEVector.h"
#include "Epetra_FECrsGraph.h"
#include "Epetra_FECrsMatrix.h"
#include "Epetra_Import.h"

// AztecOO includes
#include "AztecOO.h"
#include "AztecOO_StatusTestResNorm.h"

// ML includes
#include "ml_include.h"
#include "ml_MultiLevelPreconditioner.h"

// Ifpack includes
#include "Ifpack.h"
#include "Ifpack_ConfigDefs.h"
#include "Ifpack_AdditiveSchwarz.h"
#include "Ifpack_ILUT.h"
#include "Ifpack_IC.h"
#include "Ifpack_ICT.h"


/**************************************************************/
/*                      Types Definitions                     */
/**************************************************************/
/* Scalar types aliases */
using Scalar_d = double;
using Scalar_i = int;
using Scalar_c = std::complex<double>;

/* Ordinals and node aliases */
using LO = int;
using GO = int;
using Node = Kokkos::Compat::KokkosSerialWrapperNode;  // for CPU run

/* Tpetra type aliases */
using Tpetra_Map            = Tpetra::Map<LO, GO, Node>;
using Tpetra_CrsMatrix      = Tpetra::CrsMatrix<Scalar_d, LO, GO, Node>;
using Tpetra_BlockCrsMatrix = Tpetra::BlockCrsMatrix<Scalar_d, LO, GO, Node>;
using Tpetra_MultiVector    = Tpetra::MultiVector<Scalar_d, LO, GO, Node>; 
using Tpetra_Vector         = Tpetra::Vector<Scalar_d, LO, GO, Node>;
using Tpetra_Import         = Tpetra::Import<LO, GO, Node>;
using Tpetra_CrsGraph       = Tpetra::CrsGraph<LO, GO, Node>;
using Tpetra_Operator       = Tpetra::Operator<Scalar_d, LO, GO, Node>;

/* Belos aliases */
using Belos_LinearProblem = Belos::LinearProblem<Scalar_d, Tpetra_MultiVector, Tpetra_Operator>;
using Belos_SolverFactory = Belos::SolverFactory<Scalar_d, Tpetra_MultiVector, Tpetra_Operator>;
using Belos_StatusTestResNorm = Belos::StatusTestResNorm<Scalar_d, Tpetra_MultiVector, Tpetra_Operator>;

/* IFPACK2 preconditioner aliases */  
using Ifpack2_Preconditioner = Ifpack2::Preconditioner<Scalar_d, LO, GO, Node>;

/* MueLu preconditioner aliases */
using MueLu_Preconditioner = MueLu::Preconditioner<Scalar_d, LO, GO>;
/**************************************************************/
/*                      Macro Definitions                     */
/**************************************************************/

// Define linear solver as following naming in FSILS_struct
#define TRILINOS_CG_SOLVER 798
#define TRILINOS_GMRES_SOLVER 797
#define TRILINOS_BICGSTAB_SOLVER 795

// Define preconditioners as following naming in FSILS_struct
#define NO_PRECONDITIONER 700
#define TRILINOS_DIAGONAL_PRECONDITIONER 702
#define TRILINOS_BLOCK_JACOBI_PRECONDITIONER 703
#define TRILINOS_ILU_PRECONDITIONER 704
#define TRILINOS_ILUT_PRECONDITIONER 705
#define TRILINOS_IC_PRECONDITIONER 706
#define TRILINOS_ICT_PRECONDITIONER 707
#define TRILINOS_ML_PRECONDITIONER 708

/// @brief Initialize all Epetra types we need separate from Fortran
struct Trilinos
{
  // static Epetra_BlockMap *blockMap;
  // static Epetra_FEVector *F;
  // static Epetra_FEVbrMatrix *K;
  // static Epetra_Vector *X;
  // static Epetra_Vector *ghostX;
  // static Epetra_Import *Importer;
  // static std::vector<Epetra_FEVector*> bdryVec_list;
  // static Epetra_MpiComm *comm;
  // static Epetra_FECrsGraph *K_graph;

  static Teuchos::RCP<const Tpetra_Map> Map; 
  static Teuchos::RCP<Tpetra_MultiVector> F;
  static Teuchos::RCP<Tpetra_MultiVector> ghostF;
  static Teuchos::RCP<Tpetra_CrsMatrix> K;
  static Teuchos::RCP<Tpetra_Vector> X;
  static Teuchos::RCP<Tpetra_Vector> ghostX;
  static Teuchos::RCP<Tpetra_Import> Importer;
  static std::vector<Teuchos::RCP<Tpetra_MultiVector>> bdryVec_list;
  static Teuchos::RCP<const Teuchos::Comm<int>> comm;
  static Teuchos::RCP<Tpetra_CrsGraph> K_graph;
};

/**
 * \class TrilinosMatVec
 * \brief This class implements the pure virtual class Epetra_Operator for the
 *        AztecOO iterative solve which only uses the Apply() method to compute
 *        the matrix vector product
 */
class TrilinosMatVec: public Tpetra_Operator
{
public:

  /** Define matrix vector operation at each iteration of the linear solver
   *  adds on the coupled neuman boundary contribution to the matrix
   *
   *  \param x vector to be applied on the operator
   *  \param y result of sprase matrix vector multiplication
   */
  // int Apply(const Epetra_MultiVector &x, Epetra_MultiVector &y) const;

  /* Y = beta * Y + alpha * A^mode * X */
  void apply(const Tpetra_MultiVector& X, Tpetra_MultiVector& Y,
           Teuchos::ETransp mode = Teuchos::NO_TRANS,
           Scalar_d alpha = Teuchos::ScalarTraits< Scalar_d >::one(), 
           Scalar_d beta  = Teuchos::ScalarTraits<Scalar_d>::zero()) const override;

  /*  
    Returns the map describing the layout of the domain vector space.
    This map defines the distribution of the input vectors to the operator.
  */
  Teuchos::RCP<const Tpetra_Map> getDomainMap() const override
  {
    return Trilinos::K->getDomainMap();
  }

  /* 
    Returns the map describing the layout of the range vector space.
    This map defines the distribution of the output vectors from the operator.
  */
  Teuchos::RCP<const Tpetra_Map> getRangeMap() const override
  {
    return Trilinos::K->getRangeMap();
  }

  /** Tells whether to use the transpose of the matrix in each matrix
   * vector product */
  // int SetUseTranspose(bool use_transpose)
  // {
  //   return Trilinos::K->SetUseTranspose(use_transpose);
  // }

  // /// Computes A_inv*x
  // int ApplyInverse(const Epetra_MultiVector &X, Epetra_MultiVector &Y) const
  // {
  //   return Trilinos::K->ApplyInverse(X,Y);
  // }

  /* This handles transpose if it is needed */
  // if (mode == Teuchos::TRANS) {
  //   // Apply the transpose of the operator
  //   for (auto& bdryVec : Trilinos::bdryVec_list) {
  //     bdryVec->Multiply(alpha, X, beta, Y);
  //   }
  // } else {
  //   // Apply the operator without transpose
  //   Trilinos::K->apply();

  /// Infinity norm for global stiffness does not add in the boundary term
  /* Remove norm, if needed manually compute it */
  // double NormInf() const
  // {
  //   return Trilinos::K->NormInf();
  // }

  /// Returns a character string describing the operator
  /*  Tpetra does not support label: alternative create a string memmber
      defined in the constructor
  */
  // const char * Label() const
  // {
  //   return Trilinos::K->Label();
  // }

  // /// Returns current UseTranspose setting
  /* Tpetra handles the transpose through the Apply method */
  // bool UseTranspose() const
  // {
  //   return Trilinos::K->UseTranspose();
  // }

  // /// Returns true if this object can provide an approx Inf-norm false otherwise
  /* Tpetra does not support inf norm of matrix: compute manually if needed */
  // bool HasNormInf() const
  // {
  //   return Trilinos::K->HasNormInf();
  // }

  /// Returns pointer to Epetra_Comm communicator associated with this operator
  /* In Tpetra teh communicator is handled by Tpetra::Map no need to provide 
     a method here 
  */
  // const Epetra_Comm &Comm() const
  // {
  //   return Trilinos::K->Comm();
  // }

  /// Returns Epetra_Map object assoicated with domain of this operator
  // const Epetra_Map &OperatorDomainMap() const
  // {
  //   return Trilinos::K->OperatorDomainMap();
  // }

  /// Returns the Epetra_Map object associated with teh range of this operator
  // const Epetra_Map &OperatorRangeMap() const
  // {
  //   return Trilinos::K->OperatorRangeMap();
  // }

};// class TrilinosMatVec

//  --- Functions to be called in fortran -------------------------------------

#ifdef __cplusplus
  extern "C"
  {
#endif
  /// Give function definitions which will be called through fortran
  void trilinos_lhs_create(const int numGlobalNodes, const int numLocalNodes,
          const int numGhostAndLocalNodes, const int nnz, const Vector<int>& ltgSorted,
          const Vector<int>& ltgUnsorted, const Vector<int>& rowPtr, const Vector<int>& colInd,
          const int dof, const int cpp_index, const int proc_id, const int numCoupledNeumannBC);
/*
  void trilinos_lhs_create_(unsigned &numGlobalNodes, unsigned &numLocalNodes,
          unsigned &numGhostAndLocalNodes, unsigned &nnz, const int *ltgSorted,
          const int *ltgUnsorted, const int *rowPtr, const int *colInd,
          int &dof);
*/

  /**
   * \param v           coeff in the scalar product
   * \param isCoupledBC determines if coupled resistance BC is turned on
   */
  void trilinos_bc_create_(const std::vector<Array<double>> &v_list, bool &isCoupledBC);

  void trilinos_doassem_(int &numNodesPerElement, const int *eqN,
          const double *lK, double *lR);

  void trilinos_global_solve_(const double *Val, const double *RHS,
          double *x, const double *dirW, double &resNorm, double &initNorm,
          int &numIters, double &solverTime, double &dB, bool &converged,
          int &lsType, double &relTol, int &maxIters, int &kspace,
          int &precondType);

  void trilinos_solve_(double *x, const double *dirW, double &resNorm,
          double &initNorm, int &numIters, double &solverTime,
          double &dB, bool &converged, int &lsType, double &relTol,
          int &maxIters, int &kspace, int &precondType, bool &isFassem);

  void trilinos_lhs_free_();

#ifdef __cplusplus  /* this brace matches the one on the extern "C" line */
  }
#endif

// --- Define functions to only be called in C++ ------------------------------
// void setPreconditioner(int precondType, AztecOO &Solver);
void setPreconditioner(int precondType, Teuchos::RCP<Belos_LinearProblem>& BelosProblem);

// void setMLPrec(AztecOO &Solver);
void setMueLuPreconditioner(Teuchos::RCP<MueLu_Preconditioner>& MueLuPrec, 
  const Teuchos::RCP<Tpetra_CrsMatrix>& A)

// void setIFPACKPrec(AztecOO &Solver);

void checkDiagonalIsZero();

void constructJacobiScaling(const double *dirW,
              Tpetra_Vector &diagonal);

// --- Debugging functions ----------------------------------------------------
void printMatrixToFile();

void printRHSToFile();

void printSolutionToFile();

#endif //TRILINOS_LINEAR_SOLVER_H
