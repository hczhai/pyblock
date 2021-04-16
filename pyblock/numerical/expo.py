
import numpy as np
try:
    from expokitpy import dsexpv
except:
    pass
import os
import sys
from contextlib import contextmanager

@contextmanager
def stdout_redirected(to=os.devnull):
    fd = sys.stdout.fileno()

    def _redirect_stdout(to):
        sys.stdout.close()
        os.dup2(to.fileno(), fd)
        sys.stdout = os.fdopen(fd, 'w')

    if fd == 1:
        with os.fdopen(os.dup(fd), 'w') as old_stdout:
            with open(to, 'w') as file:
                _redirect_stdout(to=file)
            try:
                yield
            finally:
                _redirect_stdout(to=old_stdout)
    else:
        yield

def expo(a, b, beta, const_a=0.0, expo_tol=0, deflation_max_size=20):
    """Calculate exp(-beta (a + const_a)) b."""
    n = b.ref.size
    m = min(deflation_max_size, n - 1)
    v = b.ref.copy()
    wsp = np.zeros(1000 + 7 + n * (m + 2) + 5 * (m + 2) * (m + 2), dtype=float)
    iwsp = np.zeros(100 + m + 3, dtype=int)
    
    anorm = a.diag_norm()
    if anorm < 1E-10:
        anorm = 1.0

    tmpa = b.clear_copy()
    tmpb = b.clear_copy()

    icnt = [0]
    def adot(x):
        icnt[0] += 1
        tmpa.ref[:] = x[:]
        tmpb.ref[:] = 0.0
        a.apply(tmpa, tmpb)
        y = np.array(tmpb.ref.copy())
        if const_a == 0.0:
            return y
        else:
            return y + const_a * x

    with stdout_redirected():
        u, tol, iflag = dsexpv(m, -beta, v, expo_tol, anorm, wsp, iwsp, lambda x: adot(x), 0)
    
    tmpb.deallocate()
    tmpa.deallocate()

    assert iflag == 0

    nexpo = icnt[0]
    
    b.ref[:] = u[:]
    return b, nexpo

