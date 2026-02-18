"""
mpc_solver.py — Phase 3: Condensed Convex QP Solver
=====================================================
Paper: Di Carlo et al., IROS 2018  (Section IV, Eq. 27–32)

Implements the condensed QP formulation that eliminates all intermediate
state variables, leaving only the horizon-length force vector U as the
optimisation variable.

Condensed system (Eq. 27):
    X = Aqp·x₀ + Bqp·U

Objective (Eq. 28–32):
    min  ½ Uᵀ H U + gᵀ U
     U
    where
      H = 2(BqpᵀL·Bqp + K)        (Eq. 31)
      g = 2 Bqpᵀ L (Aqp·x₀ − Xref)  (Eq. 32)
      L = block-diag(Q, …, Q)   (K copies of the 13×13 state-cost diagonal)
      K = α · I_{3nk}            (force-magnitude cost)

Force constraints per foot per timestep (Eq. 22–24):
    Stance:  f_min ≤ fz ≤ f_max
             −μfz ≤ fx ≤ μfz     (friction pyramid approximation)
             −μfz ≤ fy ≤ μfz
    Swing:   fx = fy = fz = 0

Supports two solvers (auto-detected at import time):
    • quadprog  (preferred — faster, fewer dependencies)
    • osqp      (fallback)
"""

from typing import Optional, Tuple

import numpy as np

from config import NUM_LEGS, DEBUG_LEVEL, dbg
from robot_params import Go2Params, GO2

# ── Solver auto-detection ─────────────────────────────────────────────────────
try:
    import quadprog
    _SOLVER = "quadprog"
except ImportError:
    try:
        import osqp
        import scipy.sparse as sp
        _SOLVER = "osqp"
    except ImportError:
        raise ImportError(
            "No QP solver found.  Install one:\n"
            "  pip install quadprog\n"
            "  pip install osqp"
        )


class ConvexMPC:
    """
    Condensed convex model-predictive controller.

    Usage::

        mpc = ConvexMPC(K=10)
        Aqp, Bqp     = mpc.condense(Ad_list, Bd_list)
        H, g         = mpc.cost(Aqp, Bqp, x0, Xref)
        C, lb, ub    = mpc.constraints(schedule)
        U_opt        = mpc.solve(H, g, C, lb, ub)
        grf = U_opt[0:12].reshape(4, 3)   # first-timestep GRFs
    """

    NS = 13   # state dimension
    NU = 12   # input dimension  (4 feet × 3 force components)

    def __init__(self, K: int, p: Go2Params = GO2):
        self.K    = K
        self.p    = p
        self._U_prev: Optional[np.ndarray] = None
        self._solve_count = 0
        self._fail_count  = 0
        print(f"[ConvexMPC] K={K}, solver={_SOLVER}, "
              f"α={p.alpha:.1e}, f∈[{p.f_min},{p.f_max}]N, μ={p.mu}")

    # ── Step 1: Build Aqp, Bqp (Eq. 27) ──────────────────────────────────────

    def condense(self, Ad_list: list, Bd_list: list
                 ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Construct Aqp ∈ R^{K·ns × ns} and Bqp ∈ R^{K·ns × K·nu}
        such that  X = Aqp·x₀ + Bqp·U  encodes all dynamics.

        Aqp stacks powers of Ad:  [Ad¹; Ad²; … ; AdK] · x₀
        Bqp[r, c] = AdⱼAd_{j-1}···Ad_{c+1} · Bd_c   for r ≥ c
        """
        K, ns, nu = self.K, self.NS, self.NU
        Aqp = np.zeros((K * ns, ns))
        Bqp = np.zeros((K * ns, K * nu))

        # Build Aqp: cumulative product of state-transition matrices
        Phi = np.eye(ns)
        for k in range(K):
            Phi = Ad_list[k] @ Phi
            Aqp[k*ns:(k+1)*ns] = Phi

        # Build Bqp: lower-triangular block Toeplitz
        for r in range(K):
            for c in range(r + 1):
                Phi_span = np.eye(ns)
                for j in range(c + 1, r + 1):
                    Phi_span = Ad_list[j] @ Phi_span
                Bqp[r*ns:(r+1)*ns, c*nu:(c+1)*nu] = Phi_span @ Bd_list[c]

        return Aqp, Bqp

    # ── Step 2: Build H, g cost matrices (Eq. 31, 32) ────────────────────────

    def cost(self, Aqp: np.ndarray, Bqp: np.ndarray,
             x0: np.ndarray, Xref: np.ndarray
             ) -> Tuple[np.ndarray, np.ndarray]:
        """
        H = 2(BqpᵀL·Bqp + K)        (Eq. 31)
        g = 2 Bqpᵀ L (Aqp·x₀ − Xref)  (Eq. 32)

        L = block-diag(Q) repeated K times
        K = α·I  (scalar force penalty)
        """
        K, nu = self.K, self.NU
        L    = np.kron(np.eye(K), np.diag(self.p.Q))
        Kmat = self.p.alpha * np.eye(K * nu)
        BtL  = Bqp.T @ L
        H    = 2.0 * (BtL @ Bqp + Kmat)
        H    = 0.5 * (H + H.T)   # enforce exact symmetry (numerical clean-up)
        g    = 2.0 * BtL @ (Aqp @ x0 - Xref)

        if DEBUG_LEVEL >= 3:
            cond      = np.linalg.cond(H)
            bqp_norm  = np.linalg.norm(Bqp)
            err_norm  = np.linalg.norm(Aqp @ x0 - Xref)
            dbg(3, f"QP cost: cond(H)={cond:.1e}, |Bqp|={bqp_norm:.3f}, "
                   f"|Aqp·x₀−Xref|={err_norm:.3f}, |g|={np.linalg.norm(g):.3f}")
        return H, g

    # ── Step 3: Build constraint matrix (Eq. 20–24) ──────────────────────────

    def constraints(self, schedule: np.ndarray
                    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Build C, lb, ub for the per-foot force constraints over the horizon.

        schedule: (K, 4) bool — True = foot is in stance at that timestep

        Per STANCE foot → 5 rows:
            fz ∈ [f_min, f_max]
            fx − μfz ≤ 0   (and symmetric)
            fy − μfz ≤ 0   (and symmetric)
        Per SWING foot → 3 rows:
            fx = fy = fz = 0   (equality constraint)
        """
        K, nu = self.K, self.NU
        mu, fmn, fmx = self.p.mu, self.p.f_min, self.p.f_max

        n_rows = sum(
            5 if schedule[k, i] else 3
            for k in range(K) for i in range(NUM_LEGS)
        )
        C  = np.zeros((n_rows, K * nu))
        lb = np.full(n_rows, -np.inf)
        ub = np.full(n_rows,  np.inf)

        row = 0
        for k in range(K):
            u0 = k * nu
            for i in range(NUM_LEGS):
                fx_col = u0 + i*3
                fy_col = fx_col + 1
                fz_col = fx_col + 2

                if not schedule[k, i]:
                    # ── Swing: force must be exactly zero ──────────────────
                    for col in (fx_col, fy_col, fz_col):
                        C[row, col]     = 1.0
                        lb[row] = ub[row] = 0.0
                        row += 1
                else:
                    # ── Stance: normal-force bounds ────────────────────────
                    C[row, fz_col] = 1.0
                    lb[row] = fmn
                    ub[row] = fmx
                    row += 1

                    # ── Friction pyramid (Eq. 23, 24) ──────────────────────
                    # fx + μfz ≤ 0  →  ub: fx − (−μfz) ≤ 0
                    # Equivalent rows: C·u ≤ 0  with no lower bound
                    C[row, fx_col] =  1.0; C[row, fz_col] = -mu; ub[row] = 0.; row += 1
                    C[row, fx_col] = -1.0; C[row, fz_col] = -mu; ub[row] = 0.; row += 1
                    C[row, fy_col] =  1.0; C[row, fz_col] = -mu; ub[row] = 0.; row += 1
                    C[row, fy_col] = -1.0; C[row, fz_col] = -mu; ub[row] = 0.; row += 1

        return C, lb, ub

    # ── Step 4: Solve QP ──────────────────────────────────────────────────────

    def solve(self, H: np.ndarray, g: np.ndarray,
              C: np.ndarray, lb: np.ndarray, ub: np.ndarray) -> np.ndarray:
        """Dispatch to the available solver."""
        self._solve_count += 1
        if _SOLVER == "quadprog":
            return self._solve_quadprog(H, g, C, lb, ub)
        return self._solve_osqp(H, g, C, lb, ub)

    def _solve_quadprog(self, H, g, C, lb, ub) -> np.ndarray:
        """
        quadprog expects:
            min  ½xᵀGx − aᵀx
            s.t. Cᵀx ≥ b
        We split C·u ∈ [lb, ub] into lb rows and −ub rows.
        """
        n    = H.shape[0]
        r_lo = np.where(np.isfinite(lb))[0]
        r_hi = np.where(np.isfinite(ub))[0]
        Cq   = np.vstack([C[r_lo], -C[r_hi]]).T    # (n × n_ineq)
        bq   = np.concatenate([lb[r_lo], -ub[r_hi]])
        try:
            H_reg = H + 1e-7 * np.eye(n)            # small regularisation
            sol   = quadprog.solve_qp(H_reg, -g, Cq, bq)[0]
            self._U_prev = sol.copy()

            if DEBUG_LEVEL >= 3:
                fz_total = sum(sol[i*3+2] for i in range(NUM_LEGS))
                dbg(3, f"QP OK: Σfz={fz_total:.1f}N "
                       f"(weight={GO2.mass*GO2.g:.1f}N), |U|={np.linalg.norm(sol):.2f}")
            return sol
        except Exception as e:
            self._fail_count += 1
            dbg(1, f"[QP FAIL #{self._fail_count}] quadprog: {e} "
                   f"(solve #{self._solve_count})")
            return self._fallback()

    def _solve_osqp(self, H, g, C, lb, ub) -> np.ndarray:
        """OSQP interface: min ½xᵀPx + qᵀx  s.t. l ≤ Ax ≤ u."""
        n    = H.shape[0]
        H_s  = sp.csc_matrix(H + 1e-7 * np.eye(n))
        C_s  = sp.csc_matrix(C)
        prob = osqp.OSQP()
        prob.setup(H_s, g, C_s, lb, ub,
                   warm_starting=True, verbose=False,
                   eps_abs=1e-4, eps_rel=1e-4, max_iter=4000)
        if self._U_prev is not None:
            prob.warm_start(x=self._U_prev)
        res = prob.solve()
        if res.info.status not in ('solved', 'solved_inaccurate'):
            self._fail_count += 1
            dbg(1, f"[QP FAIL #{self._fail_count}] osqp: {res.info.status}")
            return self._fallback()
        self._U_prev = res.x.copy()
        return res.x

    def _fallback(self) -> np.ndarray:
        """
        Gravity-compensating fallback when QP fails.
        Sets fz = m·g / n_feet for every foot over the entire horizon.
        """
        U  = np.zeros(self.K * self.NU)
        fz = np.clip(GO2.mass * GO2.g / NUM_LEGS, self.p.f_min, self.p.f_max)
        for k in range(self.K):
            for i in range(NUM_LEGS):
                U[k * self.NU + i*3 + 2] = fz
        dbg(1, f"[QP fallback] fz per foot = {fz:.1f}N")
        return U

    # ── Diagnostics ───────────────────────────────────────────────────────────

    @property
    def stats(self) -> dict:
        return {"solves": self._solve_count, "fails": self._fail_count}