
#ifndef REV_OPERATOR_FUNCTIONS_H_
#define REV_OPERATOR_FUNCTIONS_H_

#include "StackBaseOperator.h"
#include "StackMatrix.h"
#include "StateInfo.h"
#include <boost/shared_ptr.hpp>
#include <vector>

using namespace std;
using namespace SpinAdapted;

namespace block2 {

void TensorTraceElement(const StackSparseMatrix &a, StackSparseMatrix &c,
                        const vector<boost::shared_ptr<StateInfo>> &state_info,
                        StackMatrix &cel, int cq, int cqprime, bool trace_right, double scale);

// TENSOR TRACE A x I -> C (trace_right) I x A -> C (trace_left)
void TensorTrace(const StackSparseMatrix &a, StackSparseMatrix &c,
                 const vector<boost::shared_ptr<StateInfo>> &state_info,
                 bool trace_right, double scale = 1.0);

void TensorProductElement(const StackSparseMatrix &a, const StackSparseMatrix &b, const StackSparseMatrix &c,
                          const vector<boost::shared_ptr<StateInfo>> &state_info,
                          StackMatrix &cel, int cq, int cqprime, double scale);

// TENSOR PRODUCT A x B -> C
void TensorProduct(const StackSparseMatrix &a, const StackSparseMatrix &b, const StackSparseMatrix &c,
                   const vector<boost::shared_ptr<StateInfo>> &state_info, double scale = 1.0);

// TENSOR ROTATION A -> C = T^T A T
void TensorRotate(const StackSparseMatrix &a, StackSparseMatrix &c,
                  const vector<boost::shared_ptr<StateInfo>> &state_info,
                  const vector<Matrix>& rotate_matrix);

}

#endif /* REV_OPERATOR_FUNCTIONS_H_ */