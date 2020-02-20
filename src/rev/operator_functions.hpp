
#ifndef REV_OPERATOR_FUNCTIONS_HPP_
#define REV_OPERATOR_FUNCTIONS_HPP_

#include "StackBaseOperator.h"
#include "Stackwavefunction.h"
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

// TENSOR PRODUCT A x I -> C (trace_right) I x A -> C (trace_left)
void TensorTrace(const StackSparseMatrix &a, StackSparseMatrix &c,
                 const vector<boost::shared_ptr<StateInfo>> &state_info,
                 bool trace_right, double scale = 1.0);

// TENSOR PRODUCT diag(A x I) -> diag(C) (trace_right) diag(I x A) -> diag(C) (trace_left)
// only diagonal elements of C are calculated in regular dense matrix form
void TensorTraceDiagonal(const StackSparseMatrix &a, DiagonalMatrix &c,
                         const vector<boost::shared_ptr<StateInfo>> &state_info,
                         bool trace_right, double scale);

void TensorProductElement(const StackSparseMatrix &a, const StackSparseMatrix &b, const StackSparseMatrix &c,
                          const vector<boost::shared_ptr<StateInfo>> &state_info,
                          StackMatrix &cel, int cq, int cqprime, double scale);

// TENSOR PRODUCT A x B -> C
void TensorProduct(const StackSparseMatrix &a, const StackSparseMatrix &b, StackSparseMatrix &c,
                   const vector<boost::shared_ptr<StateInfo>> &state_info, double scale = 1.0);

// PRODUCT (no kron product) A x B -> C
void Product(const StackSparseMatrix &a, const StackSparseMatrix &b, const StackSparseMatrix &c,
             const StateInfo &state_info, double scale);

// TENSOR PRODUCT diag(A x B) -> diag(C)
// only diagonal elements of C are calculated in regular dense matrix form
void TensorProductDiagonal(const StackSparseMatrix &a, const StackSparseMatrix &b, DiagonalMatrix &c,
                          const vector<boost::shared_ptr<StateInfo>> &state_info, double scale);

// TENSOR PRODUCT ACT ON STATE (A x B) C -> V [V = A C B]
void TensorProductMultiply(const StackSparseMatrix &a, const StackSparseMatrix &b,
                    const StackWavefunction &c, StackWavefunction &v,
                    const StateInfo &state_info, const SpinQuantum op_q, double scale);

// TENSOR ACT ON STATE (A x I) C -> V (trace_right) (I x A) C -> V (trace_left)
void TensorTraceMultiply(const StackSparseMatrix &a, const StackWavefunction &c,
                         StackWavefunction &v, const StateInfo &state_info,
                         bool trace_right, double scale);

// SparseMatrix ROTATION T^T A T -> C
void TensorRotate(const StackSparseMatrix &a, StackSparseMatrix &c,
                  const vector<boost::shared_ptr<StateInfo>> &state_info,
                  const vector<Matrix>& rotate_matrix, double scale);

//  SparseMatrix scale A *= scale
void TensorScale(double scale, StackSparseMatrix &a);

// SparseMatrix scale add C += scale * A
// can support a transpose
void TensorScaleAdd(double scale, const StackSparseMatrix &a, StackSparseMatrix &c,
                    const StateInfo &state_info);

// SparseMatrix scale add C += scale * A
void TensorScaleAdd(double scale, const StackSparseMatrix &a, StackSparseMatrix &c);

//  SparseMatrix dot product
double TensorDotProduct(const StackSparseMatrix &a, const StackSparseMatrix &b);

// SparseMatrix a[:] /= (e - diag[:])
void TensorPrecondition(StackSparseMatrix &a, double e, const DiagonalMatrix &diag);

}

#endif /* REV_OPERATOR_FUNCTIONS_HPP_ */