import dataclasses

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from .camera import Camera
from .face import Face


@dataclasses.dataclass(frozen=True)
class FaceModel:
    LANDMARKS: np.ndarray
    REYE_INDICES: np.ndarray
    LEYE_INDICES: np.ndarray
    MOUTH_INDICES: np.ndarray
    NOSE_INDICES: np.ndarray
    # MAP: np.ndarray
    CHIN_INDEX: int
    NOSE_INDEX: int

    def estimate_head_pose(self, face: Face, camera: Camera, mode) -> None:
        """Estimate the head pose by fitting 3D template model."""
        # If the number of the template points is small, cv2.solvePnP
        # becomes unstable, so set the default value for rvec and tvec
        # and set useExtrinsicGuess to True.
        # The default values of rvec and tvec below mean that the
        # initial estimate of the head pose is not rotated and the
        # face is in front of the camera.
        # print(f'base landmarks: {self.LANDMARKS}')
        # print(f'face landmarks: {face.landmarks}')

        rvec = np.zeros(3, dtype=np.float)
        tvec = np.array([0, 0, 1], dtype=np.float)

        if mode == 'mediapipe':
            _, rvec, tvec = cv2.solvePnP(self.LANDMARKS,
                                         face.landmarks,
                                         camera.camera_matrix,
                                         camera.dist_coefficients,
                                         rvec,
                                         tvec,
                                         useExtrinsicGuess=True,
                                         flags=cv2.SOLVEPNP_ITERATIVE)
            rot = Rotation.from_rotvec(rvec)
        elif mode == 'retina':
            points_base = []
            wflw_indices = []

            MAP = np.array([[33, 60], [133, 64], [159, 62], [145, 66], [362, 68], [263, 72], [386, 70], [374, 74], [1, 54], [2, 57], [78, 76], [308, 82], [199,16], [34, 0], [356, 32],
            [107, 37], [55,38], [70, 33], [336, 42], [285, 50], [300, 46], [66, 36], [105, 35], [63, 34], [65, 39], [52, 40], [53, 41], [296, 43], [334, 44], [293, 45],
            [295, 49], [282, 48], [283, 47], [168, 51], [197, 52], [5, 53], [98, 55], [97, 56], [326, 58], [327, 59], [0, 79], [12, 90], [37, 78], [267, 80]])

            # print(f'landmarks length: {self.LANDMARKS.shape}')
            for media_idx, wflw_idx in MAP:
                points_base.append(self.LANDMARKS[media_idx])
                wflw_indices.append(wflw_idx)   

            points_base = np.array(points_base, dtype=np.float32)

            landmarks_used = []
            for idx in wflw_indices:
                landmarks_used.append(np.array(face.landmarks[idx], dtype=np.float32))
            landmarks_used = np.array(landmarks_used)

            _, rvec, tvec = cv2.solvePnP(points_base,
                                        landmarks_used,
                                        camera.camera_matrix,
                                        camera.dist_coefficients,
                                        rvec,
                                        tvec,
                                        useExtrinsicGuess=True,
                                        flags=cv2.SOLVEPNP_ITERATIVE)
            rot = Rotation.from_rotvec(rvec)

        face.head_pose_rot = rot
        face.head_position = tvec
        face.reye.head_pose_rot = rot
        face.leye.head_pose_rot = rot

    def compute_3d_pose(self, face: Face) -> None:
        """Compute the transformed model."""
        rot = face.head_pose_rot.as_matrix()
        face.model3d = self.LANDMARKS @ rot.T + face.head_position
        # print(f'head_position: {face.head_position}')

    def compute_face_eye_centers(self, face: Face, mode: str) -> None:
        """Compute the centers of the face and eyes.

        In the case of MPIIFaceGaze, the face center is defined as the
        average coordinates of the six points at the corners of both
        eyes and the mouth. In the case of ETH-XGaze, it's defined as
        the average coordinates of the six points at the corners of both
        eyes and the nose. The eye centers are defined as the average
        coordinates of the corners of each eye.
        """
        if mode == 'ETH-XGaze':
            face.center = face.model3d[np.concatenate(
                [self.REYE_INDICES, self.LEYE_INDICES,
                 self.NOSE_INDICES])].mean(axis=0)
        else:
            face.center = face.model3d[np.concatenate(
                [self.REYE_INDICES, self.LEYE_INDICES,
                 self.MOUTH_INDICES])].mean(axis=0)
        face.reye.center = face.model3d[self.REYE_INDICES].mean(axis=0)
        face.leye.center = face.model3d[self.LEYE_INDICES].mean(axis=0)
