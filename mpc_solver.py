"""
mpc_solver.py — Terrain-Aware Condensed Convex QP Solver
=========================================================

Flat-ground behaviour is preserved when support_plane=None.

If support_plane is provided:
    • Friction cones are rotated into support frame
    • Normal force bounds applied along plane normal
"""

from typing import Optional, Tuple
import numpy as np

from config import NUM_LEGS, DEBUG_LEVEL, dbg
from robot_params import Go2Params, GO2

try:
    import quadprog
    _SOLVER = "quadprog"
except ImportError:
    try:
        import osqp
        import scipy.sparse as sp
        _SOLVER = "osqp"
    except ImportError:
        raise ImportError("Install quadprog or osqp.")


class ConvexMPC:

    NS = 13
    NU = 12

    def __init__(self, K: int, p: Go2Params = GO2):
        self.K = K
        self.p = p
        self._U_prev = None
        self._solve_count = 0
        self._fail_count = 0
        print(f"[ConvexMPC] K={K}, solver={_SOLVER}, μ={p.mu}")

    # ─────────────────────────────────────────────
    # Condensed dynamics
    # ─────────────────────────────────────────────

    def condense(self, Ad_list, Bd_list):
        K, ns, nu = self.K, self.NS, self.NU
        Aqp = np.zeros((K*ns, ns))
        Bqp = np.zeros((K*ns, K*nu))

        Phi = np.eye(ns)
        for k in range(K):
            Phi = Ad_list[k] @ Phi
            Aqp[k*ns:(k+1)*ns] = Phi

        for r in range(K):
            for c in range(r+1):
                Phi_span = np.eye(ns)
                for j in range(c+1, r+1):
                    Phi_span = Ad_list[j] @ Phi_span
                Bqp[r*ns:(r+1)*ns, c*nu:(c+1)*nu] = Phi_span @ Bd_list[c]

        return Aqp, Bqp

    # ─────────────────────────────────────────────
    # Cost
    # ─────────────────────────────────────────────

    def cost(self, Aqp, Bqp, x0, Xref, Q_override=None):

        K, nu = self.K, self.NU
        Q = Q_override if Q_override is not None else self.p.Q

        L = np.kron(np.eye(K), np.diag(Q))
        Kmat = self.p.alpha * np.eye(K*nu)

        BtL = Bqp.T @ L
        H = 2.0 * (BtL @ Bqp + Kmat)
        H = 0.5 * (H + H.T)
        g = 2.0 * BtL @ (Aqp @ x0 - Xref)

        return H, g

    # ─────────────────────────────────────────────
    # Terrain-aware constraints
    # ─────────────────────────────────────────────

    def constraints(self,
                    schedule: np.ndarray,
                    support_plane=None):

        K, nu = self.K, self.NU
        mu, fmn, fmx = self.p.mu, self.p.f_min, self.p.f_max

        n_rows = sum(
            5 if schedule[k, i] else 3
            for k in range(K)
            for i in range(NUM_LEGS)
        )

        C  = np.zeros((n_rows, K*nu))
        lb = np.full(n_rows, -np.inf)
        ub = np.full(n_rows,  np.inf)

        row = 0

        if support_plane is None:
            R_plane = np.eye(3)
        else:
            R_plane = support_plane.R

        for k in range(K):
            u0 = k * nu

            for i in range(NUM_LEGS):

                fx_col = u0 + i*3
                fy_col = fx_col + 1
                fz_col = fx_col + 2

                if not schedule[k, i]:
                    # Swing leg → zero force
                    for col in (fx_col, fy_col, fz_col):
                        C[row, col] = 1.0
                        lb[row] = ub[row] = 0.0
                        row += 1
                else:
                    # Rotate force into support frame
                    # f_plane = Rᵀ f
                    # Build transformation rows

                    # Normal component (plane z)
                    nx, ny, nz = R_plane[:,2]

                    C[row, fx_col] = nx
                    C[row, fy_col] = ny
                    C[row, fz_col] = nz
                    lb[row] = fmn
                    ub[row] = fmx
                    row += 1

                    # Tangential x constraint
                    tx = R_plane[:,0]
                    C[row, fx_col] =  tx[0]
                    C[row, fy_col] =  tx[1]
                    C[row, fz_col] =  tx[2]
                    C[row, fx_col] -= mu * nx
                    C[row, fy_col] -= mu * ny
                    C[row, fz_col] -= mu * nz
                    ub[row] = 0.0
                    row += 1

                    C[row, fx_col] = -tx[0]
                    C[row, fy_col] = -tx[1]
                    C[row, fz_col] = -tx[2]
                    C[row, fx_col] -= mu * nx
                    C[row, fy_col] -= mu * ny
                    C[row, fz_col] -= mu * nz
                    ub[row] = 0.0
                    row += 1

                    # Tangential y constraint
                    ty = R_plane[:,1]
                    C[row, fx_col] =  ty[0]
                    C[row, fy_col] =  ty[1]
                    C[row, fz_col] =  ty[2]
                    C[row, fx_col] -= mu * nx
                    C[row, fy_col] -= mu * ny
                    C[row, fz_col] -= mu * nz
                    ub[row] = 0.0
                    row += 1

                    C[row, fx_col] = -ty[0]
                    C[row, fy_col] = -ty[1]
                    C[row, fz_col] = -ty[2]
                    C[row, fx_col] -= mu * nx
                    C[row, fy_col] -= mu * ny
                    C[row, fz_col] -= mu * nz
                    ub[row] = 0.0
                    row += 1

        return C, lb, ub

    # ─────────────────────────────────────────────
    # Solve
    # ─────────────────────────────────────────────

    def solve(self, H, g, C, lb, ub):

        self._solve_count += 1

        if _SOLVER == "quadprog":
            return self._solve_quadprog(H, g, C, lb, ub)
        else:
            return self._solve_osqp(H, g, C, lb, ub)

    def _solve_quadprog(self, H, g, C, lb, ub):

        n = H.shape[0]

        r_lo = np.where(np.isfinite(lb))[0]
        r_hi = np.where(np.isfinite(ub))[0]

        Cq = np.vstack([C[r_lo], -C[r_hi]]).T
        bq = np.concatenate([lb[r_lo], -ub[r_hi]])

        try:
            sol = quadprog.solve_qp(H + 1e-7*np.eye(n), -g, Cq, bq)[0]
            self._U_prev = sol.copy()
            return sol
        except Exception as e:
            self._fail_count += 1
            dbg(1, f"[QP FAIL] {e}")
            return self._fallback()

    def _solve_osqp(self, H, g, C, lb, ub):

        n = H.shape[0]
        Hs = sp.csc_matrix(H + 1e-7*np.eye(n))
        Cs = sp.csc_matrix(C)

        prob = osqp.OSQP()
        prob.setup(Hs, g, Cs, lb, ub, verbose=False)

        if self._U_prev is not None:
            prob.warm_start(x=self._U_prev)

        res = prob.solve()

        if res.info.status not in ('solved', 'solved_inaccurate'):
            self._fail_count += 1
            dbg(1, "[OSQP FAIL]")
            return self._fallback()

        self._U_prev = res.x.copy()
        return res.x

    def _fallback(self):
        U = np.zeros(self.K * self.NU)
        fz = np.clip(GO2.mass * GO2.g / NUM_LEGS,
                     self.p.f_min,
                     self.p.f_max)
        for k in range(self.K):
            for i in range(NUM_LEGS):
                U[k*self.NU + i*3 + 2] = fz
        return U

    @property
    def stats(self):
        return {"solves": self._solve_count,
                "fails": self._fail_count}