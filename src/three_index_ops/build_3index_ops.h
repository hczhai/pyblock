/*                                                                           
Developed by Sandeep Sharma and Garnet K.-L. Chan, 2012                      
Copyright (c) 2012, Garnet K.-L. Chan                                        
                                                                             
This program is integrated in Molpro with the permission of 
Sandeep Sharma and Garnet K.-L. Chan
*/


#ifndef BUILD_3INDEX_OPS_H
#define BUILD_3INDEX_OPS_H

#include "StackBaseOperator.h"
#include "Stackspinblock.h"

namespace SpinAdapted{
namespace Three_index_ops{

//-----------------------------------------------------------------------------------------------------------------------------------------------------------

void build_3index_ops( const opTypes& optype, StackSpinBlock& big,
                       const opTypes& lhsType1, const opTypes& lhsType2,
                       const opTypes& rhsType1, const opTypes& rhsType2,
                       const std::vector<Matrix>& rotateMatrix, const StateInfo *stateinfo );


void build_3index_single_op( const opTypes& optype, const StackSpinBlock& big,
			     const opTypes& lhsType1, const opTypes& lhsType2,
			     const opTypes& rhsType1, const opTypes& rhsType2,
			     StackSparseMatrix& op);

//-----------------------------------------------------------------------------------------------------------------------------------------------------------

}
}

#endif

