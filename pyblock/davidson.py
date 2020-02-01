
import numpy as np

# general interface of Vector for Davidson algorithm
class Vector:
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
        return Vector(self.data.copy(), self.factor)
    
    def clear_copy(self):
        return Vector(np.zeros_like(self.data), self.factor)
    
    def copy_data(self, other):
        self.data = other.data.copy()
        self.factor = other.factor
    
    def dot(self, other):
        return np.dot(self.data, other.data) * self.factor * other.factor
    
    def precondition(self, ld, diag):
        assert len(diag) == len(self.data)
        for i in range(len(self.data)):
            if abs(ld - diag[i]) > 1E-12:
                self.data[i] /= ld - diag[i]
    
    def normalize(self):
        self.data = self.data / np.sqrt(np.dot(self.data, self.data))
        self.factor = 1.0
    
    def deallocate(self):
        assert self.data is not None
        self.data = None
    
    def __repr__(self):
        return repr(self.factor) + " * " + repr(self.data)

# general interface of Matrix for Davidson algorithm
class Matrix:
    def __init__(self, arr):
        self.data = arr
    
    def diag(self):
        return np.diag(self.data)
    
    def apply(self, other, result):
        result.data = np.dot(self.data, other.data)
        result.factor = other.factor

def olsen_precondition(q, c, ld, diag):
    t = c.copy()
    t.precondition(ld, diag)
    numerator = t.dot(q)
    denominator = c.dot(t)
    q += (-numerator / denominator) * c
    q.precondition(ld, diag)
    t.deallocate()

# E.R. Davidson, J. Comput. Phys. 17 (1), 87-94 (1975).
def davidson(a, b, k, max_iter=100, conv_thold=5e-6, deflation_min_size=2, deflation_max_size=20):
    assert len(b) == k
    if deflation_min_size < k:
        deflation_min_size = k
    aa = a.diag()
    for i in range(k):
        for j in range(i):
            b[i] += (-b[j].dot(b[i])) * b[j]
        b[i].normalize()
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
        atilde = np.zeros((m, m))
        for i in range(m):
            for j in range(i + 1):
                atilde[i, j] = b[i].dot(sigma[j])
                atilde[j, i] = atilde[i, j]
        ld, alpha = np.linalg.eigh(atilde)
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
        # q = sigma[ck] - b[ck] * ld[ck]
        q.copy_data(sigma[ck])
        q += (-ld[ck]) * b[ck]
        qq = q.dot(q)
#         print("%5d %5d %5d %15.8f %9.2e" % (xiter, m, ck, ld[ck], qq))
        
        # precondition
        olsen_precondition(q, b[ck], ld[ck], aa)
        
        if qq < conv_thold:
            ck += 1
            if ck == k:
                break
        else:
            if m >= deflation_max_size:
#                 print("deflating block davidson...")
                m = deflation_min_size
                msig = deflation_min_size
            for j in range(m):
                q += (-b[j].dot(q)) * b[j]
            q.normalize()
            
            if m >= len(b):
                b.append(b[0].clear_copy())
                sigma.append(sigma[0].clear_copy())
            b[m].copy_data(q)
            m += 1
        
        if xiter == max_iter:
            print("only %d converged" % ck)
    
    for i in range(len(b) - 1, k - 1, -1):
        sigma[i].deallocate()
        b[i].deallocate()
    
    q.deallocate()
    for i in range(0, k):
        sigma[i].deallocate()
    
    return ld[:ck], b[:ck]

if __name__ == "__main__":
    
    def test_7():
        a = np.array([[0,  2,  9,  2,  0,  4,  5],
                      [2,  0,  6,  5,  2,  5,  9],
                      [9,  6,  0,  4,  5,  8,  1],
                      [2,  5,  4,  0,  0,  3,  5],
                      [0,  2,  5,  0,  0,  2,  9],
                      [4,  5,  8,  3,  2,  0,  4],
                      [5,  9,  1,  5,  9,  4,  0]], dtype=float)

        b = np.identity(7, dtype=float)

        e = np.array([-16.24341216, -7.16254184, -5.12344007, -3.41825462, 0.68226548,
                      4.63978769, 26.62559552])

        ld, nb = davidson(Matrix(a), [Vector(b[0]), Vector(b[1])], 2)
        print(ld, e)
    
    def test_rand(n, k):
        a = np.zeros((n, n))
        for i in range(n):
            for j in range(i, n):
                a[i, j] = np.random.random()
                a[j, i] = a[i, j]
        ee, vv = np.linalg.eigh(a)
        
        b = [Vector(ib) for ib in np.eye(k, n)]
        
        ld, nb = davidson(Matrix(a), b, k, deflation_max_size=max(5, k + 10))
        print('std = ', ee[:k])
        print('dav = ', ld)
        print('std = ', vv[:, 0])
        print('dav = ', nb[0])
    
    test_rand(2000, 5)
