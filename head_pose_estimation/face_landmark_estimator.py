from typing import List

import dlib
import face_alignment
import face_alignment.detection.sfd
import mediapipe
import numpy as np
from omegaconf import DictConfig
import torchvision.models as models
import torch
import cv2

from common import Face
from head_pose_estimation.retina import RetinaFace_V2 as Retina
from head_pose_estimation.retina import PriorBox
from head_pose_estimation.pipnet import PIPNet
from head_pose_estimation.pipnet import get_meanface
from head_pose_estimation.retina import py_cpu_nms

class LandmarkEstimator:
    def __init__(self, config: DictConfig):
        self.device = config.device
        self.mode = config.face_detector.mode
        if self.mode == 'dlib':
            self.detector = dlib.get_frontal_face_detector()
            self.predictor = dlib.shape_predictor(
                config.face_detector.dlib_model_path)
        elif self.mode == 'face_alignment_dlib':
            self.detector = dlib.get_frontal_face_detector()
            self.predictor = face_alignment.FaceAlignment(
                face_alignment.LandmarksType._2D,
                face_detector='dlib',
                flip_input=False,
                device=config.device)
        elif self.mode == 'face_alignment_sfd':
            self.detector = face_alignment.detection.sfd.sfd_detector.SFDDetector(
                device=config.device)
            self.predictor = face_alignment.FaceAlignment(
                face_alignment.LandmarksType._2D,
                flip_input=False,
                device=config.device)
        elif self.mode == 'mediapipe':
            self.detector = mediapipe.solutions.face_mesh.FaceMesh(
                max_num_faces=config.face_detector.mediapipe_max_num_faces,
                static_image_mode=config.face_detector.
                mediapipe_static_image_mode)
        elif self.mode == 'retina':
            mbnet1 = models.mobilenet_v2(pretrained=True)
            mbnet2 = models.mobilenet_v2(pretrained=True)
            cfg = {
                'min_sizes': [[16, 32], [64, 128]],
                'steps': [8, 16],
                'variance': [0.1, 0.2],
                'clip': False,
                'loc_weight': 2.0,
                'gpu_train': False,
                'batch_size': 32,
                'ngpu': 1,
                'epoch': 250,
                'decay1': 190,
                'decay2': 220,
                'image_size': 256,
                'return_layers' : {'5':2,'10':3},
            }
            
            meanface_indices, reverse_index1, reverse_index2, max_len = get_meanface("models/face_detector/meanface.txt", 10)

            priorbox = PriorBox(cfg,image_size=(256,256))
            priors=priorbox.forward()
            prior_data = priors.data
            self.detector = Retina(cfg, prior_data, mbnet=mbnet1)#.to(self.device)

            self.detector.load_state_dict(torch.load("models/face_detector/pretrained_bboxmodel.pt",map_location=config.device),strict=False)
            self.detector.eval()

            self.landmark_detector = PIPNet(mbnet2, 10, reverse_ind1=reverse_index1, reverse_ind2=reverse_index2, max_len=max_len)#.to(self.device)
            self.landmark_detector.load_state_dict(torch.load('models/face_detector/landmark_detector.pt',map_location=config.device),strict=False)
            self.landmark_detector.eval()
        else:
            raise ValueError

    def detect_faces(self, image: np.ndarray) -> List[Face]:
        if self.mode == 'dlib':
            return self._detect_faces_dlib(image)
        elif self.mode == 'face_alignment_dlib':
            return self._detect_faces_face_alignment_dlib(image)
        elif self.mode == 'face_alignment_sfd':
            return self._detect_faces_face_alignment_sfd(image)
        elif self.mode == 'mediapipe':
            return self._detect_faces_mediapipe(image)
        elif self.mode == 'retina':
            return self._detect_faces_retina(image)
        else:
            raise ValueError

    def _detect_faces_retina(self, image: np.ndarray) -> List[Face]:
        image = cv2.resize(image, (256, 256))
        _image = image.copy()
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.astype(np.float32)
        image -= (125, 125, 125)
        image = torch.from_numpy(image)
        image = image.unsqueeze(0).permute(0,3,1,2)#.to(self.device)
        
        bbox, conf = self.detector(image)

        scores = conf
        inds=np.where(scores>0.02)
        boxes=bbox[inds]
        scores = conf[inds]
        order=scores.argsort(descending=True)
        boxes=boxes[order].detach().numpy()
        scores = scores[order].detach().numpy()
        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        keep = py_cpu_nms(dets, 0.4)
        dets = dets[keep, :]
        
        detected = [list(map(int, b)) for b in dets if b[4] > 0.9]
        faces = []

        for d in detected:
            bbox = np.array([[int(d[0] * (640/256)), int(d[1] * (480/256))], [int(d[2] * (640/256)), int(d[3] * (480/256))]], dtype=np.float)
            
            # add margins to the face image before feeding to lms net
            ori_x1 = max(int(d[0]-25.6),0)
            ori_x2 = min(int(d[2]+25.6),256)
            ori_y1 = max(int(d[1]-25.6),0)
            ori_y2 = min(int(d[3]+25.6),256)
            ori_h = ori_y2-ori_y1
            ori_w = ori_x2-ori_x1
            h = 256 - int(d[1]-25.6)
            w = 256 - int(d[0]-25.6)

            cropped = _image[ori_y1:ori_y2,ori_x1:ori_x2,:].copy()
            crp = cv2.resize(cropped,(256,256)).copy()
            cropped_img = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
            cropped_img = cv2.resize(cropped_img,(256,256))
            cropped_img = cropped_img.astype(np.float32)
            cropped_img -= (125,125,125)
            cropped_img = torch.from_numpy(cropped_img)
            cropped_img = cropped_img.unsqueeze(0).permute(0,3,1,2)#.to(self.device)
            b = self.landmark_detector(cropped_img)

            b[0::2] = ori_x1 * (640/256) + b[0::2] / 256 * ori_w * (640/256)
            b[1::2] = ori_y1 * (480/256) + b[1::2] / 256 * ori_h * (480/256)

            b = b.detach().numpy()

            landmarks = []
            for i in range(0, int(len(b)), 2):
                landmarks.append(np.array([b[i], b[i+1]], dtype=np.float))

            landmarks = np.array(landmarks, dtype=np.float)
            faces.append(Face(bbox, landmarks))

        return faces

    def _detect_faces_dlib(self, image: np.ndarray) -> List[Face]:
        bboxes = self.detector(image[:, :, ::-1], 0)
        detected = []
        for bbox in bboxes:
            predictions = self.predictor(image[:, :, ::-1], bbox)
            landmarks = np.array([(pt.x, pt.y) for pt in predictions.parts()],
                                 dtype=np.float)
            bbox = np.array([[bbox.left(), bbox.top()],
                             [bbox.right(), bbox.bottom()]],
                            dtype=np.float)
            detected.append(Face(bbox, landmarks))
        return detected

    def _detect_faces_face_alignment_dlib(self,
                                          image: np.ndarray) -> List[Face]:
        bboxes = self.detector(image[:, :, ::-1], 0)
        bboxes = [[bbox.left(),
                   bbox.top(),
                   bbox.right(),
                   bbox.bottom()] for bbox in bboxes]
        predictions = self.predictor.get_landmarks(image[:, :, ::-1],
                                                   detected_faces=bboxes)
        if predictions is None:
            predictions = []
        detected = []
        for bbox, landmarks in zip(bboxes, predictions):
            bbox = np.array(bbox, dtype=np.float).reshape(2, 2)
            detected.append(Face(bbox, landmarks))
        return detected

    def _detect_faces_face_alignment_sfd(self,
                                         image: np.ndarray) -> List[Face]:
        bboxes = self.detector.detect_from_image(image[:, :, ::-1].copy())
        bboxes = [bbox[:4] for bbox in bboxes]
        predictions = self.predictor.get_landmarks(image[:, :, ::-1],
                                                   detected_faces=bboxes)
        if predictions is None:
            predictions = []
        detected = []
        for bbox, landmarks in zip(bboxes, predictions):
            bbox = np.array(bbox, dtype=np.float).reshape(2, 2)
            detected.append(Face(bbox, landmarks))
        return detected

    def _detect_faces_mediapipe(self, image: np.ndarray) -> List[Face]:
        h, w = image.shape[:2]
        predictions = self.detector.process(image[:, :, ::-1])

        detected = []
        if predictions.multi_face_landmarks:
            for prediction in predictions.multi_face_landmarks:
                pts = np.array([(pt.x * w, pt.y * h)
                                for pt in prediction.landmark],
                               dtype=np.float64)
                bbox = np.vstack([pts.min(axis=0), pts.max(axis=0)])
                bbox = np.round(bbox).astype(np.int32)

                # print(f'pts: {pts}')
                detected.append(Face(bbox, pts))
                # print(f'bbox: {bbox}')

        return detected
