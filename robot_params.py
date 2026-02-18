"""
robot_params.py — Go2 physical parameters and MPC state dataclass

State x ∈ R¹³:  [φ, θ, ψ,  px, py, pz,  ωx, ωy, ωz,  vx, vy, vz,  −g]
                  0  1  2   3   4   5    6   7   8    9  10  11   12
"""

from dataclasses import dataclass, field
import numpy as np


@dataclass
class Go2Params:
    mass : float = 15.0
    g    : float = 9.81

    Ixx  : float = 0.0468
    Iyy  : float = 0.2447
    Izz  : float = 0.2547

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
    Q: np.ndarray = field(default_factory=lambda: np.array([
         10.,  20.,   1.,   # roll(φ), pitch(θ), yaw(ψ)
          1.,   1.,  50.,   # px, py, pz
          1.,   1.,   1.,   # ωx, ωy, ωz
          2.,   2.,   1.,   # vx, vy, vz
          0.,             # −g
    ], dtype=float))

    alpha: float = 1e-4

    mu   : float = 0.6
    f_min: float =  5.0
    f_max: float = 200.0

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
