"""
support_plane.py — Support Plane Estimation
==========================================

Computes support plane from stance feet.

Given 3+ stance feet:
  - Fit plane via least-squares
  - Compute unit normal n
  - Construct orthonormal basis:
        z_plane = n
        x_plane = projection of world x onto plane
        y_plane = z × x
"""

import numpy as np


class SupportPlane:

    def __init__(self):
        self.normal = np.array([0., 0., 1.])
        self.origin = np.zeros(3)
        self.R = np.eye(3)

    def update(self, foot_positions: np.ndarray, contacts: np.ndarray):
        """
        foot_positions : (4,3)
        contacts       : (4,) bool
        """

        pts = foot_positions[contacts]

        if pts.shape[0] < 3:
            # Not enough points → assume flat
            self.normal = np.array([0., 0., 1.])
            self.origin = pts.mean(axis=0) if pts.size else np.zeros(3)
            self.R = np.eye(3)
            return

        # Least squares plane fit
        centroid = pts.mean(axis=0)
        A = pts - centroid
        _, _, vh = np.linalg.svd(A)
        n = vh[-1]

        if n[2] < 0:
            n = -n

        n = n / np.linalg.norm(n)

        self.normal = n
        self.origin = centroid

        # Build support frame
        x_world = np.array([1., 0., 0.])
        x_plane = x_world - np.dot(x_world, n) * n
        if np.linalg.norm(x_plane) < 1e-6:
            x_plane = np.array([0., 1., 0.])
            x_plane -= np.dot(x_plane, n) * n

        x_plane /= np.linalg.norm(x_plane)
        y_plane = np.cross(n, x_plane)

        self.R = np.vstack([x_plane, y_plane, n]).T