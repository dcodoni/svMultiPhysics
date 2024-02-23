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

#include "LinearAlgebra.h"
#include "PetscLinearAlgebra.h"
#include "FsilsLinearAlgebra.h"
#include "TrilinosLinearAlgebra.h"

const std::map<std::string, consts::LinearAlgebraType> LinearAlgebra::name_to_type = {
  {"none", consts::LinearAlgebraType::none},
  {"fsils", consts::LinearAlgebraType::fsils},
  {"petsc", consts::LinearAlgebraType::petsc},
  {"trilinos", consts::LinearAlgebraType::trilinos}
};

const std::map<consts::LinearAlgebraType, std::string> LinearAlgebra::type_to_name = {
  {consts::LinearAlgebraType::none, "none"},
  {consts::LinearAlgebraType::fsils, "fsils"},
  {consts::LinearAlgebraType::petsc, "petsc"},
  {consts::LinearAlgebraType::trilinos, "trilinos"}
};

LinearAlgebra::LinearAlgebra()
{
}

LinearAlgebra* LinearAlgebraFactory::create_interface(consts::LinearAlgebraType interface_type)
{
  LinearAlgebra* interface = nullptr;

  switch (interface_type) {
    case consts::LinearAlgebraType::fsils:
      interface = new FsilsLinearAlgebra();
    break;

    case consts::LinearAlgebraType::petsc:
      interface = new PetscLinearAlgebra();
    break;

    case consts::LinearAlgebraType::trilinos:
      interface = new TrilinosLinearAlgebra();
    break;
  }

  return interface;
}
