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
Davidson diagonalization algorithm.
"""

import numpy as np

class DavidsonError(Exception):
    pass

class Vector:
    """General interface of Vector for Davidson algorithm"""
    def __init__(self, arr, factor=1.0):
        self.data = arr
        self.factor = factor
    
    def __rmul__(self, factor):
        return Vector(self.data, self.factor * factor)
    
    def __imul__(self, factor):
        self.factor *= factor
        return self
    
    def __iadd__(self, other):
        self.data = self.factor * self.data + other.factor * other.data
        self.factor = 1.0
        return self
    
    def copy(self):
        """Return a deep copy of this object."""
        return Vector(self.data.copy(), self.factor)
    
    def clear_copy(self):
        """Return a deep copy of this object, but all the matrix elements are set to zero."""
        return Vector(np.zeros_like(self.data), self.factor)
    
    def copy_data(self, other):
        """Fill the matrix elements in this object with data
        from another :class:`Vector` object."""
        self.data = other.data.copy()
        self.factor = other.factor
    
    def dot(self, other):
        """Dot product."""
        return np.dot(self.data, other.data) * self.factor * other.factor
    
    def precondition(self, ld, diag):
        """
        Apply precondition on this object.
        
        Args:
            ld : float
                Eigenvalue.
            diag : numpy.ndarray
                Diagonal elements of Hamiltonian, stored in 1D array.
        """
        assert len(diag) == len(self.data)
        for i in range(len(self.data)):
            if abs(ld - diag[i]) > 1E-12:
                self.data[i] /= ld - diag[i]
    
    def normalize(self):
        """Normalization."""
        self.data = self.data / np.sqrt(np.dot(self.data, self.data))
        self.factor = 1.0
    
    def deallocate(self):
        """Deallocate the memory associated with this object.
        This is no-op for numpy.ndarray backend used here."""
        assert self.data is not None
        self.data = None
    
    @property
    def ref(self):
        return self.data
    
    def __repr__(self):
        return repr(self.factor) + " * " + repr(self.data)

class Matrix:
    """General interface of Matrix for Davidson algorithm."""
    def __init__(self, arr):
        self.data = arr
    
    def diag(self):
        """Diagonal elements."""
        return np.diag(self.data)
    
    def diag_norm(self):
        return np.linalg.norm(self.diag())
    
    def apply(self, other, result):
        """
        Perform :math:`\\hat{H}|\\psi\\rangle`.
        
        Args:
            other : Vector
                Input vector.
            result : Vector
                Output vector.
        """
        result.data = np.dot(self.data, other.data)
        result.factor = other.factor

def olsen_precondition(q, c, ld, diag):
    """Olsen precondition."""
    t = c.copy()
    t.precondition(ld, diag)
    numerator = t.dot(q)
    denominator = c.dot(t)
    q += (-numerator / denominator) * c
    q.precondition(ld, diag)
    t.deallocate()

# E.R. Davidson, J. Comput. Phys. 17 (1), 87-94 (1975).
def davidson(a, b, k, max_iter=100, conv_thold=5e-6, deflation_min_size=2, deflation_max_size=20, iprint=False, mpi=False):
    """
    Davidson diagonalization.
    
    Args:
        a : Matrix
            The matrix to diagonalize.
        b : list(Vector)
            The initial guesses for eigenvectors.
    
    Kwargs:
        max_iter : int
            Maximal number of davidson iteration.
        conv_thold : float
            Convergence threshold for squared norm of eigenvector.
        deflation_min_size : int
            Sub-space size after deflation.
        deflation_max_size : int
            Maximal sub-space size before deflation.
        iprint : bool
            Indicate whether davidson iteration information should be printed.
    
    Returns:
        ld : list(float)
            List of eigenvalues.
        b : list(Vector)
            List of eigenvectors.
    """
    
    if mpi:
        from mpi4py import MPI
        rank = MPI.COMM_WORLD.Get_rank()
        comm = MPI.COMM_WORLD
    else:
        rank = 0
    
    if iprint and rank == 0:
        print("")
    
    assert len(b) == k
    if deflation_min_size < k:
        deflation_min_size = k
    aa = a.diag()
    if rank == 0:
        for i in range(k):
            for j in range(i):
                b[i] += (-b[j].dot(b[i])) * b[j]
            b[i].normalize()
    if mpi:
        for i in range(k):
            comm.Bcast(b[i].ref, root=0)
    sigma = [ib.clear_copy() for ib in b[:k]]
    q = b[0].clear_copy()
    l = k
    ck = 0
    msig = 0
    m = l
    xiter = 0
    while xiter < max_iter:
        xiter += 1
        for i in range(msig, m):
            a.apply(b[i], sigma[i])
            msig += 1
        if rank == 0:
            atilde = np.zeros((m, m))
            for i in range(m):
                for j in range(i + 1):
                    atilde[i, j] = b[i].dot(sigma[j])
                    atilde[j, i] = atilde[i, j]
            ld, alpha = np.linalg.eigh(atilde)
        else:
            ld = np.zeros((m, ))
            alpha = np.zeros((m, m))
        if mpi:
            comm.Bcast(ld, root=0)
            comm.Bcast(alpha, root=0)
        if rank == 0:
            # b[1:m] = np.dot(b[:], alpha[:, 1:m])
            tmp = [ib.copy() for ib in b[:m]]
            for j in range(m):
                b[j] *= alpha[j, j]
            for j in range(m):
                for i in range(m):
                    if i != j:
                        b[j] += alpha[i, j] * tmp[i]
            # sigma[1:m] = np.dot(sigma[:], alpha[:, 1:m])
            for i in range(m):
                tmp[i].copy_data(sigma[i])
            for j in range(m):
                sigma[j] *= alpha[j, j]
            for j in range(m):
                for i in range(m):
                    if i != j:
                        sigma[j] += alpha[i, j] * tmp[i]
            for i in range(m - 1, -1, -1):
                tmp[i].deallocate()
        if mpi:
            for j in range(m):
                comm.Bcast(b[j].ref, root=0)
                comm.Bcast(sigma[j].ref, root=0)
        for i in range(ck):
            q.copy_data(sigma[i])
            q += (-ld[i]) * b[i]
            qq = q.dot(q)
            if qq >= conv_thold:
                ck = i
                break
        # q = sigma[ck] - b[ck] * ld[ck]
        q.copy_data(sigma[ck])
        q += (-ld[ck]) * b[ck]
        qq = q.dot(q)
        if iprint and rank == 0:
            print("%5d %5d %5d %15.8f %9.2e" % (xiter, m, ck, ld[ck], qq))
        
        if rank == 0:
            # precondition
            olsen_precondition(q, b[ck], ld[ck], aa)
        if mpi:
            comm.Bcast(b[ck].ref, root=0)
        
        if qq < conv_thold:
            ck += 1
            if ck == k:
                break
        else:
            if m >= deflation_max_size:
                m = deflation_min_size
                msig = deflation_min_size
            if rank == 0:
                for j in range(m):
                    q += (-b[j].dot(q)) * b[j]
                q.normalize()
            if mpi:
                comm.Bcast(q.ref, root=0)
            
            if m >= len(b):
                b.append(b[0].clear_copy())
                sigma.append(sigma[0].clear_copy())
            b[m].copy_data(q)
            m += 1
        
        if xiter == max_iter:
            raise DavidsonError("Only %d converged!" % ck)
    
    for i in range(len(b) - 1, k - 1, -1):
        sigma[i].deallocate()
        b[i].deallocate()
    
    q.deallocate()
    for i in range(0, k):
        sigma[i].deallocate()
    
    return ld[:ck], b[:ck], xiter
