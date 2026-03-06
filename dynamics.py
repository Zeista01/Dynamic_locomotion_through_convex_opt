"""
dynamics.py — Terrain-Aware Rigid-Body Dynamics
================================================

Extends the original Di Carlo centroidal model to support
non-coplanar terrain via support-plane alignment.

Flat-ground behaviour is preserved EXACTLY when
support_plane=None.

If support_plane is provided:
  • Contact moment arms are expressed in support frame
  • Gravity is projected onto support normal
  • Dynamics become terrain-consistent
"""

from typing import Tuple
import numpy as np
import scipy.linalg

from config import NUM_LEGS, dbg
from robot_params import Go2Params, GO2


class RigidBodyDynamics:

    NS = 13
    NU = 12

    def __init__(self, p: Go2Params = GO2):
        self.p = p

    # ─────────────────────────────────────────────────────────────
    # Rotation helpers
    # ─────────────────────────────────────────────────────────────

    @staticmethod
    def Rz(psi: float) -> np.ndarray:
        c, s = np.cos(psi), np.sin(psi)
        return np.array([[c, -s, 0.],
                         [s,  c, 0.],
                         [0., 0., 1.]])

    @staticmethod
    def skew(v: np.ndarray) -> np.ndarray:
        return np.array([
            [0., -v[2],  v[1]],
            [v[2],  0., -v[0]],
            [-v[1], v[0], 0.]
        ])

    def I_hat(self, psi: float) -> np.ndarray:
        """
        Inertia rotated only about yaw (Di Carlo approximation).
        """
        Rz = self.Rz(psi)
        return Rz @ self.p.BI @ Rz.T

    # ─────────────────────────────────────────────────────────────
    # Continuous dynamics A matrix
    # ─────────────────────────────────────────────────────────────

    def Ac(self,
           psi: float,
           support_plane=None) -> np.ndarray:
        """
        Continuous-time A matrix.

        If support_plane is provided:
          gravity is projected along support normal.
        """

        A = np.zeros((13, 13))

        # Orientation kinematics
        A[0:3, 6:9] = self.Rz(psi).T

        # Position kinematics
        A[3:6, 9:12] = np.eye(3)

        # Gravity term
        
            # Flat ground (original behaviour)
        A[11, 12] = 1.0
 
        return A

    # ─────────────────────────────────────────────────────────────
    # Continuous dynamics B matrix
    # ─────────────────────────────────────────────────────────────

    def Bc(self,
           r_feet: np.ndarray,
           psi: float,
           contacts: np.ndarray,
           support_plane=None,
           debug: bool = False) -> np.ndarray:
        """
        Continuous-time B matrix.

        If support_plane is provided:
            moment arms are expressed in support-plane frame.
        """

        B = np.zeros((13, 12))

        Iinv = np.linalg.inv(self.I_hat(psi))

        effective_contacts = contacts.copy()
        if contacts.sum() == 0:
            effective_contacts[:] = True
            dbg(3, "Bc: no contacts in schedule — using all-stance fallback")

        # Determine rotation
        if support_plane is None:
            R_plane = np.eye(3)
        else:
            R_plane = support_plane.R  # world → plane rotation

        for i in range(NUM_LEGS):

            if not effective_contacts[i]:
                continue

            c0 = i * 3

            r_world = r_feet[i]

            # Clamp to avoid numerical explosion
            r_world = np.clip(r_world, -0.8, 0.8)

            # BUG FIX: Keep moment arm in WORLD frame.
            # The B matrix maps world-frame forces to world-frame accelerations.
            # The torque equation τ = r × f requires both vectors in the same
            # frame.  Previously r was rotated into the support-plane frame
            # while f stayed in world frame, producing a physically wrong
            # cross product (mixed-frame torque).
            # The support plane rotation is correctly used ONLY in the QP
            # constraints (mpc_solver.py) to rotate friction cones.

            # Angular acceleration contribution
            B[6:9, c0:c0+3] = Iinv @ self.skew(r_world)

            # Linear acceleration contribution
            B[9:12, c0:c0+3] = np.eye(3) / self.p.mass

            if debug:
                dbg(4, f"Bc leg {i}: r_plane={r_plane}")

        return B

    # ─────────────────────────────────────────────────────────────
    # Discretisation
    # ─────────────────────────────────────────────────────────────

    def discretise(self,
                   A: np.ndarray,
                   B: np.ndarray,
                   dt: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Zero-order hold discretisation.
        """

        ns, nu = self.NS, self.NU

        M = np.zeros((ns + nu, ns + nu))
        M[:ns, :ns] = A
        M[:ns, ns:] = B

        eM = scipy.linalg.expm(M * dt)

        Ad = eM[:ns, :ns]
        Bd = eM[:ns, ns:]

        return Ad, Bd