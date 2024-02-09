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

#include "FsilsLinearAlgebra.h"
#include "fsils_api.hpp"
#include <iostream>

// Include FSILS-dependent data structures and functions. 
//
//#ifdef USE_PETSC
//#include "petsc_impl.cpp"

// If PETSc is not used then define FsilsImpl with noop methods.
//
//#else
//class FsilsLinearAlgebra::FsilsImpl {
  //public:
    //FsilsImpl(){};
    //void initialize(ComMod& com_mod) {};
    //void solve(ComMod& com_mod, eqType& lEq, const Vector<int>& incL, const Vector<double>& res) {};
//};
//#endif

class FsilsLinearAlgebra::FsilsImpl {
  public:
    FsilsImpl(){};
    void initialize(ComMod& com_mod) {};
    void solve(ComMod& com_mod, eqType& lEq, const Vector<int>& incL, const Vector<double>& res) {};
};

/////////////////////////////////////////////////////////////////
//             F s i l s L i n e a r A l g e b r a             //
/////////////////////////////////////////////////////////////////
// The following methods implement the LinearAlgebra interface.

FsilsLinearAlgebra::FsilsLinearAlgebra()
{
  std::cout << "[FsilsLinearAlgebra] ---------- FsilsLinearAlgebra ---------- " << std::endl;
  impl = new FsilsLinearAlgebra::FsilsImpl();
  interface_type = LinearAlgebraType::fsils; 
}

void FsilsLinearAlgebra::initialize(ComMod& com_mod)
{
  std::cout << "[FsilsLinearAlgebra] ---------- initialize ---------- " << std::endl;
  //impl->initialize(com_mod);
}

void FsilsLinearAlgebra::solve(ComMod& com_mod, eqType& lEq, const Vector<int>& incL, const Vector<double>& res)
{
  std::cout << "[FsilsLinearAlgebra] ---------- solve ---------- " << std::endl;
  auto& lhs = com_mod.lhs;
  int dof = com_mod.dof;
  auto& R = com_mod.R;      // Residual vector
  auto& Val = com_mod.Val;  // LHS matrix

  fsi_linear_solver::fsils_solve(lhs, lEq.FSILS, dof, R, Val, lEq.ls.PREC_Type, incL, res);
}

