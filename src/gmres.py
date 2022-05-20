import numpy as np
import cupy as cp
from cupyx.scipy import sparse
from cupyx.scipy.sparse import linalg


def gmres(A, b, tol=1e-8, restart=20, maxiter=10):

    n = A.shape[0]
    x = sparse.csr_matrix(b.shape, dtype=A.dtype)
    matvec = A.matvec

    if n == 0:
        return cp.empty_like(b), 0
    b_norm = linalg.norm(b)
    if b_norm == 0:
        return b, 0

    atol = tol * float(b_norm)

    if maxiter is None:
        maxiter = n * 10
    if restart is None:
        restart = 3
    restart = min(restart, n)

    H = np.zeros((restart+1, restart), dtype=A.dtype, order='F')
    e = np.zeros((restart + 1,), dtype=A.dtype)

    iters = 0
    r = b
    while True:
        r_norm = linalg.norm(r)
        if r_norm <= atol or iters >= maxiter:
            break
        v = r / r_norm
        V = dict()
        V[0] = v
        e[0] = r_norm

        # Arnoldi iteration
        for j in range(restart):
            u = matvec(V[j])
            for i in range(j+1):
                vi = V[i]
                uvi = u.multiply(vi)
                uvis = uvi.sum()
                H[i, j] = uvis
            for i in range(j+1):
                u = u - H[i, j] * V[i]
            hjj = linalg.norm(u)
            H[j+1, j] = hjj
            V[j+1] = u / H[j+1, j]

        ret = np.linalg.lstsq(cp.asnumpy(H), e, rcond=-1)
        y = cp.array(ret[0])
        for i in range(restart):
           x += V[i]*y[i]
        iters += restart

        r = b - matvec(x, linalg.norm(x))

    info = 0
    if iters == maxiter and not (r_norm <= atol):
        info = iters
    return x, info
