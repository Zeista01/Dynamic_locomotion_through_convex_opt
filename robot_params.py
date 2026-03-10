"""
robot_params.py — Go2 physical parameters and MPC state dataclass

State x ∈ R¹³:  [φ, θ, ψ,  px, py, pz,  ωx, ωy, ωz,  vx, vy, vz,  −g]
                  0  1  2   3   4   5    6   7   8    9  10  11   12
"""

from dataclasses import dataclass, field
import numpy as np


@dataclass
class Go2Params:
    mass : float = 15.206408
    g    : float = 9.81

    # Effective trot inertia: between trunk-only and full composite.
    # Trunk-only: [0.107, 0.098, 0.025] — underestimates by 2.8x/4.9x/18.7x.
    # Full composite (standing): [0.300, 0.481, 0.458] — overestimates because
    # swing legs are retracted during trot, reducing their moment contribution.
    # Tuned to ~70% of composite for pitch/roll, empirically validated:
    Ixx  : float = 0.220
    Iyy  : float = 0.312
    Izz  : float = 0.100

    # State order: [φ, θ, ψ,  px, py, pz,  ωx, ωy, ωz,  vx, vy, vz,  −g]
    #               0  1  2   3   4   5    6   7   8    9  10  11   12
    #
    # FIX: Q weights now EXACTLY match the CasADi reference (centroidal_mpc.py):
    #   COST_MATRIX_Q = diag([1,1,50,10,20,1,2,2,1,1,1,1])
    #   order in reference: [px,py,pz,roll,pitch,yaw,vx,vy,vz,wx,wy,wz]
    #
    # Mapped to our 13-state [φ,θ,ψ, px,py,pz, ωx,ωy,ωz, vx,vy,vz, -g]:
    #   roll=10, pitch=20, yaw=1, px=1, py=1, pz=50, wx=1,wy=1,wz=1, vx=2,vy=2,vz=1
    #
    # Previous values (roll=100,pitch=200,pz=500,vx=100) were 10-50x too large.
    # Those caused the MPC to generate aggressive corrective forces that
    # destabilised the trot within 2 gait cycles (confirmed from debug log).
    # State order: [φ, θ, ψ,  px, py, pz,  ωx, ωy, ωz,  vx, vy, vz,  −g]
    # Based on CasADi reference (centroidal_mpc.py).
    # Pitch drift (~5 deg at vx=0.4) is a known limitation of the SRBD model:
    # the linearised dynamics don't capture leg-mass pitch coupling during
    # diagonal stance.  Higher Q_pitch (35, 60) was tested but creates
    # positive feedback through force-pitch coupling → faster divergence.
    # The current weights are the empirically-stable sweet spot.
    # State order: [φ, θ, ψ,  px, py, pz,  ωx, ωy, ωz,  vx, vy, vz,  −g]
    # With corrected composite inertia (Iyy=0.312), the MPC generates
    # adequate pitch-correcting forces with moderate Q weights.
    # FIX: conservative yaw 1→2, vy 3→5, ωz 1→2 to reduce drift
    # without destabilising the trot (larger jumps crashed at t=6).
    # FIX: vx 8→15, vy 5→8 to prevent velocity surges during trot startup.
    # At vx=8, speed overshoot to 0.25 m/s (from cmd=0.05) caused roll crash.
    # At vx=15: penalty for 0.20 error = 0.60 (was 0.32) — 2x stronger.
    Q: np.ndarray = field(default_factory=lambda: np.array([
          20., 55., 2.,
            1.,  1., 100.,
            2.,  12.,  2.,
           15.,   8.,  5.,
            0.
    ], dtype=float))

    alpha: float = 5e-4

    mu   : float = 0.7
    f_min: float =  5.0
    f_max: float = 300.0

    @property
    def BI(self) -> np.ndarray:
        return np.diag([self.Ixx, self.Iyy, self.Izz])


GO2 = Go2Params()


@dataclass
class RobotState:
    """
    Unified state container.
    MPC 13-state: [φ, θ, ψ, px, py, pz, ωx, ωy, ωz, vx, vy, vz, −g]
    """
    euler   : np.ndarray = field(default_factory=lambda: np.zeros(3))
    pos     : np.ndarray = field(default_factory=lambda: np.zeros(3))
    omega   : np.ndarray = field(default_factory=lambda: np.zeros(3))
    vel     : np.ndarray = field(default_factory=lambda: np.zeros(3))
    R       : np.ndarray = field(default_factory=lambda: np.eye(3))
    foot_pos: np.ndarray = field(default_factory=lambda: np.zeros((4, 3)))
    r_feet  : np.ndarray = field(default_factory=lambda: np.zeros((4, 3)))
    contact : np.ndarray = field(default_factory=lambda: np.zeros(4, dtype=bool))
    cf      : np.ndarray = field(default_factory=lambda: np.zeros(4))
    q       : np.ndarray = field(default_factory=lambda: np.zeros(12))
    dq      : np.ndarray = field(default_factory=lambda: np.zeros(12))
    t       : float = 0.0

    def mpc_x(self) -> np.ndarray:
        return np.concatenate([
            self.euler,
            self.pos,
            self.omega,
            self.vel,
            np.array([-9.81]),
        ])

    def __str__(self) -> str:
        from config import LEG_NAMES
        r, p, y = np.degrees(self.euler)
        ct = "  ".join(
            f"{n}:{'Y' if v else 'N'}"
            for n, v in zip(LEG_NAMES, self.contact)
        )
        return (
            f"t={self.t:7.3f}s | "
            f"xyz=[{self.pos[0]:+.3f} {self.pos[1]:+.3f} {self.pos[2]:+.3f}] | "
            f"rpy=[{r:+.1f} {p:+.1f} {y:+.1f}]° | "
            f"vel=[{self.vel[0]:+.2f} {self.vel[1]:+.2f} {self.vel[2]:+.2f}] | "
            f"{ct}"
        )