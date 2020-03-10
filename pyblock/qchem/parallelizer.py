#
#    pyblock: Spin-adapted quantum chemistry DMRG in MPO language (based on Block C++ code)
#    Copyright (C) 2019-2020 Huanchen Zhai
#
#    Block 1.5.3: density matrix renormalization group (DMRG) algorithm for quantum chemistry
#    Developed by Sandeep Sharma and Garnet K.-L. Chan, 2012
#    Copyright (C) 2012 Garnet K.-L. Chan
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""
MPI parallelization.
"""

from .operator import OpNames, OpElement, OpString, OpSum
from .simplifier import OpCollection, OpShell
from .fcidump import TInt
from block.operator import StackSparseMatrix, Wavefunction
import contextlib
import numpy as np
from mpi4py import MPI

comm = MPI.COMM_WORLD
mpi_rank = comm.Get_rank()
mpi_size = comm.Get_size()


class ParaOpCollection(OpCollection):
    def __init__(self, uniq, linked=None, partial=None, collect=None, broadcast=None, bcast_all=False):
        super().__init__(uniq, linked)
        partial = sorted(partial.items(), key=lambda x: x[0]) if partial is not None else []
        self.uniq_list.extend(partial)
        self.collect = sorted(collect, key=lambda x: x[0])[::-1] if collect is not None else []
        self.broadcast = broadcast if broadcast is not None else []
        self.bcast_all = bcast_all
    
    @contextlib.contextmanager
    def __call__(self):
        with super().__call__() as (uniq, new_ops):
            yield uniq, new_ops
            allo_ops = []
            deallo_ops = []
            for op, owner in self.collect:
                
                max_size = 0
                if new_ops[op] == 0:
                    ccomm = comm.Split(1, mpi_rank + 1)
                    max_size = comm.allreduce(0, op=MPI.MAX)
                elif mpi_rank == owner:
                    ccomm = comm.Split(0, 0)
                    max_size = comm.allreduce(ccomm.Get_size(), op=MPI.MAX)
                else:
                    ccomm = comm.Split(0, mpi_rank + 1)
                    max_size = comm.allreduce(ccomm.Get_size(), op=MPI.MAX)
                
                if max_size != 0:
                    
                    if mpi_rank == owner:
                        kmat = None
                        for i in range(0, mpi_size):
                            if i != mpi_rank:
                                kk = comm.recv(source=i, tag=11)
                                if kk is not None:
                                    kmat = kk
                        
                        if new_ops[op] == 0:
                            new_ops[op] = StackSparseMatrix()
                            new_ops[op].deep_clear_copy(kmat)
                            allo_ops.append(op)
                            reloc = True
                        else:
                            reloc = False
                        for i in range(0, mpi_size):
                            if i != mpi_rank:
                                comm.send(reloc, dest=i, tag=12)
                        
                        if reloc:
                            ccomm.Free()
                            ccomm = comm.Split(0, 0)
                        
                    else:
                        if ccomm.Get_rank() == 0 and new_ops[op] != 0:
                            assert isinstance(new_ops[op], StackSparseMatrix)
                            comm.send(new_ops[op], dest=owner, tag=11)
                        else:
                            comm.send(None, dest=owner, tag=11)
                        
                        reloc = comm.recv(source=owner, tag=12)
                        if reloc:
                            ccomm.Free()
                            if new_ops[op] != 0:
                                ccomm = comm.Split(0, mpi_rank + 1)
                            else:
                                ccomm = comm.Split(1, mpi_rank + 1)

                if new_ops[op] != 0:
                    if mpi_rank == owner:
                        ccomm.Reduce(MPI.IN_PLACE, [new_ops[op].ref, MPI.DOUBLE], op=MPI.SUM, root=0)
                    else:
                        ccomm.Reduce([new_ops[op].ref, MPI.DOUBLE], None, op=MPI.SUM, root=0)
                        if not self.bcast_all:
                            deallo_ops.append(op)
                
                ccomm.Free()
                if not self.bcast_all and mpi_rank != owner and new_ops[op] == 0:
                    del new_ops[op]
            
            if len(deallo_ops) != 0:
                tmp = []
                for op in allo_ops[::-1]:
                    tmp.append((new_ops[op].total_memory, np.array(new_ops[op].ref[:])))
                    new_ops[op].deallocate()
                for op in deallo_ops:
                    assert not isinstance(new_ops[op], Wavefunction)
                    new_ops[op].deallocate()
                    if mpi_rank != owner:
                        del new_ops[op]
                for op in allo_ops:
                    m, data = tmp.pop()
                    new_ops[op].allocate_memory(m)
                    new_ops[op].ref[:] = data
            
            for op, owner in self.broadcast:
                comm.Bcast(new_ops[op].ref, root=owner)


class ParaProperty:
    def __init__(self, owner, repeated, partial):
        self.owner = owner
        self.repeated = repeated
        self.partial = partial
    
    @property
    def avail(self):
        return self.owner == mpi_rank or self.repeated


class ParaRule:
    def __init__(self, size=mpi_size):
        self.size = mpi_size
    
    def __call__(self, op):
        if op.name in [OpNames.C, OpNames.D]:
            return ParaProperty(0, True, False)
        elif op.name in [OpNames.R, OpNames.RD]:
            return ParaProperty(op.site_index[0] % (self.size - 1) + 1, False, True)
        elif op.name in [OpNames.I]:
            return ParaProperty(0, True, False)
        elif op.name in [OpNames.H]:
            return ParaProperty(0, False, True)
        else:
            return ParaProperty(TInt.find_index(*op.site_index[:2]) % (self.size - 1) + 1, False, False)


class ParaRule:
    def __init__(self, size=mpi_size):
        self.size = mpi_size
    
    def __call__(self, op):
        if op.name in [OpNames.C, OpNames.D]:
            return ParaProperty(0, True, False)
        elif op.name in [OpNames.R, OpNames.RD]:
            return ParaProperty(op.site_index[0] % self.size, False, True)
        elif op.name in [OpNames.I]:
            return ParaProperty(0, True, False)
        elif op.name in [OpNames.H]:
            return ParaProperty(0, False, True)
        else:
            return ParaProperty(TInt.find_index(*op.site_index[:2]) % self.size, False, False)
        
class Parallelizer:
    def __init__(self, rule, rank=mpi_rank):
        self.rule = rule
        self.op_map = {}
        self.rank = rank
        st = np.random.get_state()
        st = comm.bcast(st, root=0)
        np.random.set_state(st)
    
    def _is_expr_local(self, op, expr):
        if expr == 0:
            return True
        elif isinstance(expr, OpString):
            owner = self.op_map[op].owner
            for xop in expr.ops:
                if xop not in self.op_map:
                    self.op_map[xop] = self.rule(xop)
            if all(self.op_map[xop].owner == owner or self.op_map[xop].repeated for xop in expr.ops):
                return True
            else:
                return False
        else:
            assert isinstance(expr, OpSum)
            owner = self.op_map[op].owner
            p_strings = []
            for x in expr.strings:
                for xop in x.ops:
                    if xop not in self.op_map:
                        self.op_map[xop] = self.rule(xop)
                if all(self.op_map[xop].owner == owner or self.op_map[xop].repeated for xop in x.ops):
                    p_strings.append(x)
            if len(p_strings) == len(expr.strings):
                return True
            else:
                return False
    
    def parallelize(self, op_coll, do_partial=False, bcast_all=False):
        uniq, linked = op_coll.uniq, op_coll.linked
        p_uniq = {}
        p_uniq_partial = {}
        p_linked = []
        p_collect = []
        p_broadcast = []
        for op, expr in uniq.items():
            if op not in self.op_map:
                self.op_map[op] = self.rule(op)
            if self.op_map[op].partial and do_partial:
                if self._is_expr_local(op, expr):
                    if self.op_map[op].owner == self.rank:
                        p_uniq[op] = expr
                    if bcast_all:
                        if self.op_map[op].owner != self.rank:
                            p_uniq[op] = 0
                        p_broadcast.append((op, self.op_map[op].owner))
                else:
                    assert isinstance(expr, OpSum) or isinstance(expr, OpString)
                    if isinstance(expr, OpString):
                        expr = OpSum([expr])
                    p_strings = []
                    for x in expr.strings:
                        for xop in x.ops:
                            if xop not in self.op_map:
                                self.op_map[xop] = self.rule(xop)
                        if all(self.op_map[xop].avail for xop in x.ops):
                            p_strings.append(x)
                    lp = comm.allreduce(len(p_strings), op=MPI.SUM)
                    assert lp == len(expr.strings)
                    if self.op_map[op].owner == self.rank:
                        p_uniq[op] = OpSum(p_strings)
                    else:
                        p_uniq_partial[op] = OpSum(p_strings)
                    p_collect.append((op, self.op_map[op].owner))
                    if bcast_all:
                        p_broadcast.append((op, self.op_map[op].owner))
            elif self.op_map[op].repeated:
                if self.op_map[op].owner == self.rank:
                    p_uniq[op] = expr if expr != 0 else 0
                else:
                    p_uniq[op] = OpShell(expr) if expr != 0 else 0
                p_broadcast.append((op, self.op_map[op].owner))
            elif self.op_map[op].owner == self.rank:
                p_uniq[op] = expr
        for op, expr, link in linked:
            if op not in self.op_map:
                self.op_map[op] = self.rule(op)
            if self.op_map[op].avail:
                p_linked.append((op, expr, link))
        return ParaOpCollection(p_uniq, p_linked, p_uniq_partial, p_collect, p_broadcast, bcast_all=bcast_all)
