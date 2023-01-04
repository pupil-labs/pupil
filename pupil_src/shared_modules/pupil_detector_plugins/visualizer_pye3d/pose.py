"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""
import cv2
import numpy as np


class PosedObject:
    def __init__(self, pose=np.eye(4), extrinsics=None, children=(), parents=()):

        self.parents = parents
        self.children = children
        self._pose = np.eye(
            4
        )  # Needed during initialization for first call of pose.setter

        if type(pose) in [np.ndarray, list]:
            self.pose = np.array(pose)
        else:
            self.pose = self.pose_from_extrinsics(extrinsics)

    @property
    def pose(self):
        return self._pose.copy()

    @pose.setter
    def pose(self, new_pose):
        for child in self.children:
            child.pose = new_pose @ np.linalg.inv(self.pose) @ child.pose
        self._pose = new_pose

    def translate(self, translation):
        pose = self.pose.copy()
        pose[:3, 3] += translation
        self.pose = pose

    def rotate(self, rotation):
        pose = self.pose.copy()
        pose[:3, :3] = rotation @ pose[:3, :3]
        self.pose = pose

    @staticmethod
    def extrinsics_from_pose(pose):
        rot = pose[:3, :3]
        trans = pose[:3, 3]
        extrinsics = np.eye(4)
        extrinsics[:3, :3] = rot.T
        extrinsics[:3, 3] = -rot.T @ trans
        return extrinsics

    @staticmethod
    def pose_from_extrinsics(extrinsics):
        rot = extrinsics[:3, :3]
        trans = extrinsics[:3, 3]
        pose = np.eye(4)
        pose[:3, :3] = rot.T
        pose[:3, 3] = -rot.T @ trans
        return pose

    @property
    def tvec(self):
        return self._pose[:3, 3]

    @tvec.setter
    def tvec(self, new_tvec):
        pose = self.pose
        pose[:3, 3] = new_tvec
        self.pose = pose

    @property
    def rmat(self):
        return self._pose[:3, :3]

    @rmat.setter
    def rmat(self, new_rmat):
        pose = self.pose
        pose[:3, :3] = new_rmat
        self.pose = pose

    @property
    def rvec(self):
        return cv2.Rodrigues(self.rmat)[0]

    @rvec.setter
    def rvec(self, new_rvec):
        self._pose[:3, :3] = cv2.Rodrigues(new_rvec)[0]

    @property
    def extrinsics(self):
        return self.extrinsics_from_pose(self.pose)

    @extrinsics.setter
    def extrinsics(self, new_extrinsics):
        self.pose = self.pose_from_extrinsics(new_extrinsics)
