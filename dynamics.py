"""
dynamics.py — Phase 2: Simplified Rigid-Body Dynamics
======================================================
Paper: Di Carlo et al., IROS 2018  (Section III, Eq. 8–17)

This module implements the 13-state linear time-varying (LTV) model
that the convex MPC uses for prediction.

Key approximations made (following the paper exactly):
  1. Robot modelled as a single rigid body (leg mass ignored, ~10% of total)
  2. Translational:   p̈ = Σfᵢ / m − g                      (Eq. 5)
  3. Rotational:      İω ≈ Î·ω̇   (precession / nutation dropped)  (Eq. 13)
  4. Inertia approx:  Î = Rz(ψ)·BI·Rz(ψ)ᵀ   (ignores roll/pitch) (Eq. 15)
  5. Euler rate:      Θ̇ ≈ Rz(ψ)ᵀ·ω           (small roll/pitch)   (Eq. 12)

State x ∈ R¹³:  [φ, θ, ψ,  px, py, pz,  ωx, ωy, ωz,  vx, vy, vz,  −g]
Input u ∈ R¹²:  [f₁ₓ f₁ᵧ f₁z | f₂ₓ f₂ᵧ f₂z | f₃ₓ f₃ᵧ f₃z | f₄ₓ f₄ᵧ f₄z]

Continuous dynamics:  ẋ = Ac(ψ)·x + Bc(r₁,...,r₄,ψ)·u   (Eq. 17)
Discrete dynamics:   x[n+1] = Ad·x[n] + Bd[n]·u[n]       (Eq. 26)
  via zero-order hold (ZOH) matrix exponential             (Eq. 25)
"""

from typing import Tuple

import numpy as np
import scipy.linalg

from config import NUM_LEGS, dbg
from robot_params import Go2Params, GO2


class RigidBodyDynamics:
    """
    Builds the Ac and Bc matrices (Eq. 16/17) and discretises them (Eq. 25).

    Usage::

        dyn = RigidBodyDynamics()
        A_c = dyn.Ac(psi)
        B_c = dyn.Bc(r_feet, psi, contacts)
        Ad, Bd = dyn.discretise(A_c, B_c, dt)
    """

    NS = 13   # number of states
    NU = 12   # number of inputs  (4 feet × 3 force components)

    def __init__(self, p: Go2Params = GO2):
        self.p = p

    # ── Static helpers ─────────────────────────────────────────────────────────

    @staticmethod
    def Rz(psi: float) -> np.ndarray:
        """3×3 rotation matrix about Z-axis by angle ψ."""
        c, s = np.cos(psi), np.sin(psi)
        return np.array([[c, -s, 0.],
                         [s,  c, 0.],
                         [0., 0., 1.]])

    @staticmethod
    def skew(v: np.ndarray) -> np.ndarray:
        """
        Skew-symmetric matrix such that  skew(v) @ w = v × w.
        Used to build the rotational part of Bc (torque = r × f = skew(r)·f).
        """
        return np.array([
            [ 0.,    -v[2],  v[1]],
            [ v[2],  0.,    -v[0]],
            [-v[1],  v[0],  0.  ],
        ])

    def I_hat(self, psi: float) -> np.ndarray:
        """
        Approximate world-frame inertia (Eq. 15):
            Î = Rz(ψ)·BI·Rz(ψ)ᵀ
        Valid for small roll and pitch angles.
        """
        Rz = self.Rz(psi)
        return Rz @ self.p.BI @ Rz.T

    # ── State-space matrices ───────────────────────────────────────────────────

    def Ac(self, psi: float) -> np.ndarray:
        """
        13×13 continuous-time A matrix (Eq. 16).

        Non-zero blocks:
          A[0:3, 6:9]  = Rz(ψ)ᵀ        ← Θ̇ = Rz(ψ)ᵀ·ω   (Eq. 12)
          A[3:6, 9:12] = I₃             ← ṗ = v
          A[11, 12]    = 1              ← v̇z = ... + state[12] = ... − g
        """
        A = np.zeros((13, 13))
        A[0:3, 6:9]  = self.Rz(psi).T   # orientation kinematics (Eq. 12)
        A[3:6, 9:12] = np.eye(3)         # position kinematics
        A[11, 12]    = 1.0               # gravity enters vz equation
        return A

    def Bc(self, r_feet: np.ndarray, psi: float,
            contacts: np.ndarray, debug: bool = False) -> np.ndarray:
        """
        13×12 continuous-time B matrix (Eq. 16).

        For each foot i in stance:
          B[6:9,  i*3:i*3+3] = Î⁻¹ · skew(rᵢ)   ← angular acceleration
          B[9:12, i*3:i*3+3] = I₃ / m             ← linear  acceleration

        FIX (BUG 3): If NO feet are in stance, fall back to all-stance so
        that Bc is never identically zero (which would make Bqp = 0 and
        break the QP entirely).
        """
        B    = np.zeros((13, 12))
        Iinv = np.linalg.inv(self.I_hat(psi))

        effective_contacts = contacts.copy()
        if contacts.sum() == 0:
            effective_contacts[:] = True
            dbg(3, "Bc: no contacts in schedule — using all-stance fallback")

        for i in range(NUM_LEGS):
            if not effective_contacts[i]:
                continue
            c0 = i * 3
            r  = r_feet[i]

            # Clamp r to prevent MPC blow-up if the robot is in a weird pose
            r_clamped = np.clip(r, -0.8, 0.8)

            B[6:9,  c0:c0+3] = Iinv @ self.skew(r_clamped)
            B[9:12, c0:c0+3] = np.eye(3) / self.p.mass

            if debug:
                dbg(4, f"  Bc leg {i}: r={r_clamped}, "
                       f"|Iinv@skew|="
                       f"{np.linalg.norm(Iinv @ self.skew(r_clamped)):.3f}")
        return B

    # ── Zero-Order Hold Discretisation ────────────────────────────────────────

    def discretise(self, A: np.ndarray, B: np.ndarray,
                   dt: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Zero-order hold (ZOH) discretisation via matrix exponential (Eq. 25).

        Constructs the (ns+nu) × (ns+nu) extended matrix:
            M = [[A, B],
                 [0, 0]]
        then  e^{M·dt} = [[Ad, Bd],
                          [0,   I]]

        This gives the exact discrete-time equivalent when the input u
        is held constant over the interval [t, t+dt].
        """
        ns, nu = self.NS, self.NU
        M = np.zeros((ns + nu, ns + nu))
        M[:ns, :ns] = A
        M[:ns, ns:] = B
        eM = scipy.linalg.expm(M * dt)
        return eM[:ns, :ns], eM[:ns, ns:]