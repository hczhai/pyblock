
#include "rev/operator_functions.hpp"
#include "SpinQuantum.h"
#include "MatrixBLAS.h"
#include "couplingCoeffs.h"
#include "global.h"
#include "newmat.h"
#include <cmath>
#include <vector>
#include <iostream>

#define TINY 1.e-20

using namespace std;
using namespace SpinAdapted;

namespace block2 {

// if trace_right == true:
// contract state_info[0] with a; trace state_info[1]; output state_info[2]
// if trace_right == false:
// trace state_info[0]; contract state_info[1] with a; output state_info[2]
// if forward : state_info :: 0 =  left site, 1 = physical, 2 = currect site
// if backward: state_info :: 0 = physical, 1 = right site, 2 = currect site
//
// actual output will be in repr of direct product of state_info[0] and state_info[1]
// but in direct sum spin space, but before truncation after collection
// since it is already collected, the quantum numbers are already sorted in c repr
// so the rotation matrix is the map from collected quantum numbers to a subset of it
// with the same order
// further rotation is required to obtain repr in state_info[2]
// assuming state_info[2] is collected
void TensorTraceElement(const StackSparseMatrix &a, StackSparseMatrix &c,
                        const vector<boost::shared_ptr<StateInfo>> &state_info,
                        StackMatrix &cel, int cq, int cqprime, bool trace_right,
                        double scale) {
    
    if (fabs(scale) < TINY)
        return;

    int aq, aqprime, bq, bqprime, bstates;
    
    const StateInfo *ls = state_info[0].get();
    const StateInfo *rs = state_info[1].get();
    const StateInfo *cs = state_info[2].get();
    
    const char conjC = trace_right ? 'n' : 't';
    const std::vector<int> oldToNewI = cs->oldToNewState.at(cq);
    const std::vector<int> oldToNewJ = cs->oldToNewState.at(cqprime);
    
    int rowstride = 0, colstride = 0;

    for (int oldi = 0; oldi < oldToNewI.size(); oldi++) {
        colstride = 0;
        for (int oldj = 0; oldj < oldToNewJ.size(); oldj++) {
            if (conjC == 'n') {
                aq = cs->leftUnMapQuanta[oldToNewI[oldi]];
                aqprime = cs->leftUnMapQuanta[oldToNewJ[oldj]];
                bq = cs->rightUnMapQuanta[oldToNewI[oldi]];
                bqprime = cs->rightUnMapQuanta[oldToNewJ[oldj]];
                bstates = rs->getquantastates(bq); // bq == bqprime, which is traced (right: 1)
            } else {
                aq = cs->rightUnMapQuanta[oldToNewI[oldi]];
                aqprime = cs->rightUnMapQuanta[oldToNewJ[oldj]];
                bq = cs->leftUnMapQuanta[oldToNewI[oldi]];
                bqprime = cs->leftUnMapQuanta[oldToNewJ[oldj]];
                bstates = ls->getquantastates(bq); // bq == bqprime, which is traced (left: 0)
            }

            if (a.allowed(aq, aqprime) && (bq == bqprime)) {
                DiagonalMatrix unitMatrix(bstates);
                unitMatrix = 1.;

                Matrix unity(bstates, bstates);
                unity = unitMatrix;

                if (conjC == 'n') {
                    double scaleb = dmrginp.get_ninej()(
                        ls->quanta[aqprime].get_s().getirrep(),
                        rs->quanta[bqprime].get_s().getirrep(),
                        cs->quanta[cqprime].get_s().getirrep(),
                        a.get_spin().getirrep(), 0, c.get_spin().getirrep(),
                        ls->quanta[aq].get_s().getirrep(),
                        rs->quanta[bq].get_s().getirrep(),
                        cs->quanta[cq].get_s().getirrep());

                    scaleb *= Symmetry::spatial_ninej(
                        ls->quanta[aqprime].get_symm().getirrep(),
                        rs->quanta[bqprime].get_symm().getirrep(),
                        cs->quanta[cqprime].get_symm().getirrep(),
                        a.get_symm().getirrep(), 0, c.get_symm().getirrep(),
                        ls->quanta[aq].get_symm().getirrep(),
                        rs->quanta[bq].get_symm().getirrep(),
                        cs->quanta[cq].get_symm().getirrep());
                    
                    // no fermion check for trace right A x I(bigger site index)

                    MatrixTensorProduct(a.operator_element(aq, aqprime),
                                        a.conjugacy(), scale, unity, 'n',
                                        scaleb, cel, rowstride, colstride);
                } else {
                    double scaleb = dmrginp.get_ninej()(
                        ls->quanta[bqprime].get_s().getirrep(),
                        rs->quanta[aqprime].get_s().getirrep(),
                        cs->quanta[cqprime].get_s().getirrep(), 0,
                        a.get_spin().getirrep(), c.get_spin().getirrep(),
                        ls->quanta[bq].get_s().getirrep(),
                        rs->quanta[aq].get_s().getirrep(),
                        cs->quanta[cq].get_s().getirrep());
                    scaleb *= Symmetry::spatial_ninej(
                        ls->quanta[bqprime].get_symm().getirrep(),
                        rs->quanta[aqprime].get_symm().getirrep(),
                        cs->quanta[cqprime].get_symm().getirrep(),
                        0, a.get_symm().getirrep(), c.get_symm().getirrep(),
                        ls->quanta[bq].get_symm().getirrep(),
                        rs->quanta[aq].get_symm().getirrep(),
                        cs->quanta[cq].get_symm().getirrep());
                    
                    // fermion check for trace left I(smaller site index) x A
                    
                    if (a.get_fermion() &&
                        
                        // defined in SpinQuantum
                        IsFermion(ls->quanta[bqprime]))
                        scaleb *= -1.;

                    MatrixTensorProduct(
                        unity, 'n', scaleb, a.operator_element(aq, aqprime),
                        a.conjugacy(), scale, cel, rowstride, colstride);
                }
            }
            colstride += cs->unCollectedStateInfo->quantaStates[oldToNewJ[oldj]];
        }
        rowstride += cs->unCollectedStateInfo->quantaStates[oldToNewI[oldi]];
    }
}

void TensorTrace(const StackSparseMatrix &a, StackSparseMatrix &c,
                 const vector<boost::shared_ptr<StateInfo>> &state_info,
                 bool trace_right, double scale) {
    
    if (fabs(scale) < TINY)
        return;
    
    assert(a.get_initialised() && c.get_initialised());

    std::vector<std::pair<std::pair<int, int>, StackMatrix>> &nonZeroBlocks =
        c.get_nonZeroBlocks();

    int quanta_thrds = dmrginp.quanta_thrds();
#pragma omp parallel for schedule(dynamic) num_threads(quanta_thrds)
    for (int index = 0; index < nonZeroBlocks.size(); index++) {
        int cq = nonZeroBlocks[index].first.first,
            cqprime = nonZeroBlocks[index].first.second;
        TensorTraceElement(a, c, state_info,
            nonZeroBlocks[index].second, cq, cqprime, trace_right, scale);
    }
    
}

void TensorProductElement(const StackSparseMatrix &a, const StackSparseMatrix &b, const StackSparseMatrix &c,
                          const vector<boost::shared_ptr<StateInfo>> &state_info,
                          StackMatrix &cel, int cq, int cqprime, double scale) {
    
    if (fabs(scale) < TINY)
        return;

    const StateInfo *ketstateinfo = state_info[2].get(),
                    *brastateinfo = state_info[2].get();

    const std::vector<int> &oldToNewI = brastateinfo->oldToNewState.at(cq);
    const std::vector<int> &oldToNewJ = ketstateinfo->oldToNewState.at(cqprime);

    const char conjC = 'n';

    const StateInfo *lbraS = state_info[0].get(),
                    *rbraS = state_info[1].get();
    const StateInfo *lketS = state_info[0].get(),
                    *rketS = state_info[1].get();
    int rowstride = 0, colstride = 0;

    int aq, aqprime, bq, bqprime;

    for (int oldi = 0; oldi < oldToNewI.size(); oldi++) {
        colstride = 0;
        for (int oldj = 0; oldj < oldToNewJ.size(); oldj++) {
            aq = brastateinfo->leftUnMapQuanta[oldToNewI[oldi]];
            aqprime = ketstateinfo->leftUnMapQuanta[oldToNewJ[oldj]];
            bq = brastateinfo->rightUnMapQuanta[oldToNewI[oldi]];
            bqprime = ketstateinfo->rightUnMapQuanta[oldToNewJ[oldj]];

            double scaleA = scale;
            double scaleB = 1.0;
            
            if (a.allowed(aq, aqprime) && b.allowed(bq, bqprime)) {
                scaleB = dmrginp.get_ninej()(
                    lketS->quanta[aqprime].get_s().getirrep(),
                    rketS->quanta[bqprime].get_s().getirrep(),
                    ketstateinfo->quanta[cqprime].get_s().getirrep(),
                    a.get_spin().getirrep(), b.get_spin().getirrep(),
                    c.get_spin().getirrep(),
                    lbraS->quanta[aq].get_s().getirrep(),
                    rbraS->quanta[bq].get_s().getirrep(),
                    brastateinfo->quanta[cq].get_s().getirrep());
                scaleB *= Symmetry::spatial_ninej(
                    lketS->quanta[aqprime].get_symm().getirrep(),
                    rketS->quanta[bqprime].get_symm().getirrep(),
                    ketstateinfo->quanta[cqprime].get_symm().getirrep(),
                    a.get_symm().getirrep(), b.get_symm().getirrep(),
                    c.get_symm().getirrep(),
                    lbraS->quanta[aq].get_symm().getirrep(),
                    rbraS->quanta[bq].get_symm().getirrep(),
                    brastateinfo->quanta[cq].get_symm().getirrep());
                scaleB *= b.get_scaling(rbraS->quanta[bq],
                                        rketS->quanta[bqprime]);
                scaleA *= a.get_scaling(lbraS->quanta[aq],
                                        lketS->quanta[aqprime]);
                if (b.get_fermion() &&
                    IsFermion(lbraS->quanta[aqprime]))
                    scaleB *= -1;
                
                MatrixTensorProduct(
                    a.operator_element(aq, aqprime), a.conjugacy(),
                    scaleA, b.operator_element(bq, bqprime),
                    b.conjugacy(), scaleB, cel, rowstride, colstride);
            }
            colstride += ketstateinfo->unCollectedStateInfo->quantaStates[oldToNewJ[oldj]];
        }
        rowstride += brastateinfo->unCollectedStateInfo->quantaStates[oldToNewI[oldi]];
    }
}

void TensorProduct(const StackSparseMatrix &a, const StackSparseMatrix &b, StackSparseMatrix &c,
                   const vector<boost::shared_ptr<StateInfo>> &state_info, double scale) {
    
    if (fabs(scale) < TINY)
        return;
    
    assert(a.get_initialised() && b.get_initialised() && c.get_initialised());

    std::vector<std::pair<std::pair<int, int>, StackMatrix>> &nonZeroBlocks =
        c.get_nonZeroBlocks();

    int quanta_thrds = dmrginp.quanta_thrds();
#pragma omp parallel for schedule(dynamic) num_threads(quanta_thrds)
    for (int index = 0; index < nonZeroBlocks.size(); index++) {
        int cq = nonZeroBlocks[index].first.first,
            cqprime = nonZeroBlocks[index].first.second;
        TensorProductElement(a, b, c, state_info, 
            nonZeroBlocks[index].second, cq, cqprime, scale);
    }

}
    
void TensorRotate(const StackSparseMatrix &a, StackSparseMatrix &c,
                  const vector<boost::shared_ptr<StateInfo>> &state_info,
                  const vector<Matrix>& rotate_matrix) {
    
    const StateInfo *olds = state_info[0].get(), *news = state_info[1].get();
    
    assert(a.get_initialised() && c.get_initialised());

    std::vector<std::pair<std::pair<int, int>, StackMatrix>> &nonZeroBlocks =
        c.get_nonZeroBlocks();
    
    vector<int> new_to_old_map;
    for (int old_q = 0; old_q < rotate_matrix.size(); ++old_q)
        if (rotate_matrix[old_q].Ncols() != 0)
            new_to_old_map.push_back(old_q);
    
    int quanta_thrds = dmrginp.quanta_thrds();
#pragma omp parallel for schedule(dynamic) num_threads(quanta_thrds)
    for (int index = 0; index < nonZeroBlocks.size(); index++) {
        int cq = nonZeroBlocks[index].first.first,
            cqprime = nonZeroBlocks[index].first.second;
        int q = new_to_old_map[cq],
            qprime = new_to_old_map[cqprime];
        
        MatrixRotate(rotate_matrix[q], a.operator_element(q, qprime),
            rotate_matrix[qprime], nonZeroBlocks[index].second);
        
    }
    
}
    
} // namespace block2
