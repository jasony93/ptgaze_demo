import logging
from typing import List

import numpy as np
import torch
import cv2
from omegaconf import DictConfig
from torchvision.utils import save_image
import random

from common import Camera, Face, FacePartsName
from head_pose_estimation import HeadPoseNormalizer, LandmarkEstimator
from models import create_model
from models import Model
from transforms import create_transform
from utils import get_3d_face_model

logger = logging.getLogger(__name__)


class GazeEstimator:
    EYE_KEYS = [FacePartsName.REYE, FacePartsName.LEYE]

    def __init__(self, config: DictConfig):
        self._config = config

        self._face_model3d = get_3d_face_model(config)

        self.camera = Camera(config.gaze_estimator.camera_params)
        self._normalized_camera = Camera(
            config.gaze_estimator.normalized_camera_params)

        self._landmark_estimator = LandmarkEstimator(config)
        self._head_pose_normalizer = HeadPoseNormalizer(
            self.camera, self._normalized_camera,
            self._config.gaze_estimator.normalized_camera_distance)
        self._gaze_estimation_model = self._load_model()
        # print(self._gaze_estimation_model)
        # torch.save(self._gaze_estimation_model, r'F:\DMS\pytorch_mpiigaze_demo\ptgaze\models\eth-xgaze_resnet18_whole.pth')
        self._transform = create_transform(config)
        self.img_idx = 0

    def _load_model(self) -> torch.nn.Module:
        model = create_model(self._config)
        checkpoint = torch.load(self._config.gaze_estimator.checkpoint,
                                map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        # model.load_state_dict(checkpoint)
        # model = Model()
        # model.load_from_checkpoint("models/epoch=0014.ckpt")
        # model.load_from_checkpoint(checkpoint)
        # model.load_state_dict(checkpoint)
        model.to(torch.device(self._config.device))
        model.eval()
        return model

    def detect_faces(self, image: np.ndarray) -> List[Face]:
        return self._landmark_estimator.detect_faces(image)

    def estimate_gaze(self, image: np.ndarray, face: Face) -> None:
        self._face_model3d.estimate_head_pose(face, self.camera, self._config.face_detector.mode)
        self._face_model3d.compute_3d_pose(face)
        self._face_model3d.compute_face_eye_centers(face, self._config.mode)

        if self._config.mode == 'MPIIGaze':
            for key in self.EYE_KEYS:
                eye = getattr(face, key.name.lower())
                self._head_pose_normalizer.normalize(image, eye)
            self._run_mpiigaze_model(face)
        elif self._config.mode == 'MPIIFaceGaze':
            self._head_pose_normalizer.normalize(image, face)
            self._run_mpiifacegaze_model(face)
        elif self._config.mode == 'ETH-XGaze':
            self._head_pose_normalizer.normalize(image, face)
            self._run_ethxgaze_model(face)
        else:
            raise ValueError

    @torch.no_grad()
    def _run_mpiigaze_model(self, face: Face) -> None:
        images = []
        head_poses = []
        for key in self.EYE_KEYS:
            eye = getattr(face, key.name.lower())
            image = eye.normalized_image
            normalized_head_pose = eye.normalized_head_rot2d
            if key == FacePartsName.REYE:
                image = image[:, ::-1].copy()
                normalized_head_pose *= np.array([1, -1])
            image = self._transform(image)
            images.append(image)
            head_poses.append(normalized_head_pose)
        images = torch.stack(images)
        head_poses = np.array(head_poses).astype(np.float32)
        head_poses = torch.from_numpy(head_poses)

        device = torch.device(self._config.device)
        images = images.to(device)
        head_poses = head_poses.to(device)
        predictions = self._gaze_estimation_model(images, head_poses)
        predictions = predictions.cpu().numpy()

        for i, key in enumerate(self.EYE_KEYS):
            eye = getattr(face, key.name.lower())
            eye.normalized_gaze_angles = predictions[i]
            if key == FacePartsName.REYE:
                eye.normalized_gaze_angles *= np.array([1, -1])
            eye.angle_to_vector()
            eye.denormalize_gaze_vector()

    @torch.no_grad()
    def _run_mpiifacegaze_model(self, face: Face) -> None:
        image = self._transform(face.normalized_image).unsqueeze(0)
        # img_np = face.normalized_image.cpu().detach().numpy().squeeze(0).transpose(1,2,0)
        # print(img_np.shape)
        # gray_img = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY) * 255
        # cv2.imwrite(f'data/tmp/gray/{random.randint(0,9999)}.jpg', gray_img)

        # print(f'image shape: {image.shape}  face.normalized image shape: {face.normalized_image.shape}  gray_img shape: {gray_img.shape}')
        device = torch.device(self._config.device)
        image = image.to(device)
        prediction = self._gaze_estimation_model(image)
        prediction = prediction.cpu().numpy()

        face.normalized_gaze_angles = prediction[0]
        face.angle_to_vector()
        face.denormalize_gaze_vector()

    @torch.no_grad()
    def _run_ethxgaze_model(self, face: Face) -> None:
        image = self._transform(face.normalized_image).unsqueeze(0)
        # print(image)

        # print(f'image.shape: {image[0].shape}')
        # if self.img_idx % 3 == 0:
        #     save_image(image, f'F://DMS/pytorch_mpiigaze_demo/ptgaze/data/tmp/image_{self.img_idx}.png')
        # self.img_idx += 1

        # print('_run_ethxgaze_model')

        device = torch.device(self._config.device)
        image = image.to(device)
        prediction = self._gaze_estimation_model(image)
        prediction = prediction.cpu().numpy()

        # print(f'prediction: {prediction}')

        face.normalized_gaze_angles = prediction[0]
        face.angle_to_vector()
        face.denormalize_gaze_vector()
