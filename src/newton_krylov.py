import cupy as cp
from cupyx.scipy.sparse import linalg
import sys
from gmres import gmres

def maxnorm(x):
    return linalg.norm(x)

class TerminationCondition:

    def __init__(self, f_tol=10e-13, iter=None, norm=maxnorm):

        self.f_tol = f_tol
        self.norm = norm
        self.iter = iter

        self.f0_norm = None
        self.iteration = 0

    def check(self, f):
        self.iteration += 1
        f_norm = self.norm(f)

        if self.f0_norm is None:
            self.f0_norm = f_norm

        if f_norm == 0:
            return 1

        if self.iter is not None:
            return 2 * (self.iteration > self.iter)

        return int((f_norm <= self.f_tol and f_norm <= self.f0_norm))

class KrylovJacobian:

    def __init__(self):
        self.rdiff = cp.finfo(cp.float64).eps ** (1./2)


    def _update_diff_step(self):
        mx = self.x0.max()
        mf = self.f0.max()
        self.omega = self.rdiff * max(1, mx) / max(1, mf)


    def matvec(self, v,nv=1):

        sc = self.omega / nv
        xn = (self.x0 + sc * v).reshape(self.en * self.k, self.en)
        r = (self.func(xn) - self.f0) / sc
        return r

    def solve(self, rhs, tol=10e-8):
        sol, info = gmres(self.op, rhs, tol=tol)
        return sol

    def update(self, x, f):
        self.x0 = x
        self.f0 = f
        self._update_diff_step()


    def setup(self, x, f, func, en, k):

        self.func = func
        self.shape = (f.shape[0], x.shape[0])
        self.dtype = f.dtype

        self.x0 = x
        self.f0 = f
        self.op = self
        self.en = en
        self.k = k

        self._update_diff_step()


def nonlin_solve(F, x0, en, k, iter=None, verbose=False,
                 maxiter=None, f_tol=None, full_output=False, raise_exception=True):
    
    tol_norm = maxnorm
    
    condition = TerminationCondition(f_tol=f_tol, iter=iter, norm=tol_norm)

    func = F
    x = x0

    Fx = func(x0)
    Fx_norm = linalg.norm(Fx)
    jacobian = KrylovJacobian()
    jacobian.setup(x.copy(), Fx, func, en,k)

    if maxiter is None:
        if iter is not None:
            maxiter = iter + 1
        else:
            maxiter = 100*(x.size+1)

    eta = 1e-3 #change

    for n in range(maxiter):
        status = condition.check(Fx)
        if status:
            break

        # The tolerance, as computed for scipy.sparse.linalg.* routines
        tol = min(eta, eta*Fx_norm)
        dx = -jacobian.solve(Fx, tol=tol)

        if linalg.norm(dx) == 0:
            raise ValueError("Jacobian inversion yielded zero vector. "
                             "This indicates a bug in the Jacobian "
                             "approximation.")

        s = 1.0
        x = x + s*dx
        Fx = func(x.reshape(en*k, en))
        Fx_norm_new = cp.linalg.norm(Fx)

        jacobian.update(x.copy(), Fx)
        Fx_norm = Fx_norm_new

        # Print status
        if verbose:
            sys.stdout.write("%d:  |F(x)| = %g; step %g\n" % (n, tol_norm(Fx), s))
            sys.stdout.flush()
    else:
        if raise_exception:
            status = 3
        else:
            status = 2

    if full_output:
        info = {'nit': condition.iteration,
                'fun': Fx,
                'status': status,
                'success': status == 1,
                'message': {1: 'A solution was found at the specified '
                               'tolerance.',
                            2: 'The maximum number of iterations allowed '
                               'has been reached.',
                            3: 'NoConvergence (with exception)'
                               'has been reached.'
                            }[status]
                }
        return x, info
    else:
        return x