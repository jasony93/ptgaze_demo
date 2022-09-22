import datetime
import logging
import pathlib
from typing import Optional
import time
import matplotlib.pyplot as plt

import cv2
import numpy as np
from omegaconf import DictConfig

from common import Face, FacePartsName, Visualizer
from gaze_estimator import GazeEstimator
from utils import get_3d_face_model
from pydub import AudioSegment
from pydub.playback import play

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Demo:
    QUIT_KEYS = {27, ord('q')}

    def __init__(self, config: DictConfig):
        self.config = config
        self.gaze_estimator = GazeEstimator(config)
        face_model_3d = get_3d_face_model(config)
        if self.config.face_detector.mode == 'mediapipe':
            self.visualizer = Visualizer(self.gaze_estimator.camera,
                                        face_model_3d.NOSE_INDEX)
        elif self.config.face_detector.mode == 'retina':
            self.visualizer = Visualizer(self.gaze_estimator.camera,
                                        54)

        self.cap = self._create_capture()
        self.output_dir = self._create_output_dir()
        self.writer = self._create_video_writer()

        self.stop = False
        self.show_bbox = self.config.demo.show_bbox
        self.show_head_pose = self.config.demo.show_head_pose
        self.show_landmarks = self.config.demo.show_landmarks
        self.show_normalized_image = self.config.demo.show_normalized_image
        self.show_template_model = self.config.demo.show_template_model
        self.pitch_array = []
        self.yaw_array = []
        self.gaze_area_array = []
        self.frame_count = 0
        self.start_time = time.time()
        self.end_time = 0

    def run(self) -> None:
        if self.config.demo.use_camera or self.config.demo.video_path:
            self._run_on_video()
        elif self.config.demo.image_path:
            self._run_on_image()
        else:
            raise ValueError

    def _run_on_image(self):
        image = cv2.imread(self.config.demo.image_path)
        self._process_image(image)
        if self.config.demo.display_on_screen:
            while True:
                key_pressed = self._wait_key()
                if self.stop:
                    break
                if key_pressed:
                    self._process_image(image)
                cv2.imshow('image', self.visualizer.image)
        if self.config.demo.output_dir:
            name = pathlib.Path(self.config.demo.image_path).name
            output_path = pathlib.Path(self.config.demo.output_dir) / name
            cv2.imwrite(output_path.as_posix(), self.visualizer.image)

    def analyze(self):
        fig, axs = plt.subplots(1,2)
        for i, ax in enumerate(axs.flatten()):
            if i == 0:
                ax.plot(self.yaw_array, self.pitch_array, "k.", markersize=5)
                ax.set_xlim([-60, 60])
                ax.set_ylim([-60, 60])
                ax.set_title("Scatter plot")
            else:
                x_bins = np.linspace(-60, 60, 100)
                y_bins = np.linspace(-60, 60, 100)
                plt.hist2d(self.yaw_array, self.pitch_array, bins=[x_bins, y_bins])
                plt.savefig('result/analyze/WIN_20220902_16_03_00_Pro_2.png', dpi=300)
                # plt.show()

    def _run_on_video(self) -> None:
        logger.info('run on video')
        while True:
            if self.config.demo.display_on_screen:
                self._wait_key()
                if self.stop:
                    if self.config.demo.gaze_analyzer:
                        self.analyze()
                    break

            ok, frame = self.cap.read()
            if not ok:
                break

            self._process_image(frame)

            if self.config.demo.display_on_screen:
                cv2.imshow('frame', self.visualizer.image)

        if self.config.demo.gaze_analyzer:
            self.analyze()

        self.cap.release()
        if self.writer:
            self.writer.release()

    def _process_image(self, image) -> None:

        # print(f'image shape: {image.shape}')
        
        undistorted = cv2.undistort(
            image, self.gaze_estimator.camera.camera_matrix,
            self.gaze_estimator.camera.dist_coefficients)
        _image = image
        
        #uncomment below lines to use grayscale
        # _image = cv2.cvtColor(undistorted, cv2.COLOR_BGR2GRAY)
        # _image = cv2.cvtColor(_image, cv2.COLOR_GRAY2BGR)
        # print(undistorted.shape)
        # undistorted = image

        self.visualizer.set_image(_image.copy())
        start = time.time()
        faces = self.gaze_estimator.detect_faces(_image)
        # print(faces)

        # start = time.time()
        for face in faces:
            self.gaze_estimator.estimate_gaze(_image, face)
            self._draw_face_bbox(face)
            self._draw_head_pose(face)
            self._draw_landmarks(face)
            self._draw_face_template_model(face)
            self._draw_gaze_vector(face)
            self._display_normalized_image(face)
        
        end = time.time()
        # print(f'inference time: {end - start}')

        if self.config.demo.use_camera:
            # self.visualizer.image = self.visualizer.image[:, ::-1]
            self.visualizer.image = self.visualizer.image[:, :]
        if self.writer:
            self.writer.write(self.visualizer.image)

    def _create_capture(self) -> Optional[cv2.VideoCapture]:
        if self.config.demo.image_path:
            return None
        if self.config.demo.use_camera:
            cap = cv2.VideoCapture(self.config.demo.video_device)
        elif self.config.demo.video_path:
            cap = cv2.VideoCapture(self.config.demo.video_path)
        else:
            raise ValueError
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.gaze_estimator.camera.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.gaze_estimator.camera.height)
        return cap

    def _create_output_dir(self) -> Optional[pathlib.Path]:
        if not self.config.demo.output_dir:
            return
        output_dir = pathlib.Path(self.config.demo.output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
        return output_dir

    @staticmethod
    def _create_timestamp() -> str:
        dt = datetime.datetime.now()
        return dt.strftime('%Y%m%d_%H%M%S')

    def _create_video_writer(self) -> Optional[cv2.VideoWriter]:
        if self.config.demo.image_path:
            return None
        if not self.output_dir:
            return None
        ext = self.config.demo.output_file_extension
        if ext == 'mp4':
            fourcc = cv2.VideoWriter_fourcc(*'H264')
        elif ext == 'avi':
            fourcc = cv2.VideoWriter_fourcc(*'PIM1')
        else:
            raise ValueError
        if self.config.demo.use_camera:
            output_name = f'{self._create_timestamp()}.{ext}'
        elif self.config.demo.video_path:
            name = pathlib.Path(self.config.demo.video_path).stem
            output_name = f'{name}.{ext}'
        else:
            raise ValueError
        output_path = self.output_dir / output_name
        # print(self.gaze_estimator.camera.width, self.gaze_estimator.camera.height)
        if self.config.demo.use_camera:
            # print(self.config.demo.use_camera)
            print(self.gaze_estimator.camera.width,
                                    self.gaze_estimator.camera.height)
            writer = cv2.VideoWriter(output_path.as_posix(), fourcc, 30,
                                    (self.gaze_estimator.camera.width,
                                    self.gaze_estimator.camera.height))
            # writer = cv2.VideoWriter(output_path.as_posix(), fourcc, 30,
            #                         (int(self.config.demo.width),
            #                         int(self.config.demo.height)))
        else:
            # Have to set frame width and height accordingly
            writer = cv2.VideoWriter(output_path.as_posix(), fourcc, 30,
                                    (int(self.config.demo.width),
                                    int(self.config.demo.height)))
        if writer is None:
            raise RuntimeError
        return writer

    def _wait_key(self) -> bool:
        key = cv2.waitKey(self.config.demo.wait_time) & 0xff
        if key in self.QUIT_KEYS:
            self.stop = True
        elif key == ord('b'):
            self.show_bbox = not self.show_bbox
        elif key == ord('l'):
            self.show_landmarks = not self.show_landmarks
        elif key == ord('h'):
            self.show_head_pose = not self.show_head_pose
        elif key == ord('n'):
            self.show_normalized_image = not self.show_normalized_image
        elif key == ord('t'):
            self.show_template_model = not self.show_template_model
        else:
            return False
        return True

    def _draw_face_bbox(self, face: Face) -> None:
        if not self.show_bbox:
            return
        self.visualizer.draw_bbox(face.bbox)

    def _draw_head_pose(self, face: Face) -> None:
        if not self.show_head_pose:
            return
        # Draw the axes of the model coordinate system
        length = self.config.demo.head_pose_axis_length
        self.visualizer.draw_model_axes(face, length, lw=2)

        euler_angles = face.head_pose_rot.as_euler('XYZ', degrees=True)
        pitch, yaw, roll = face.change_coordinate_system(euler_angles)
        # print('logger print')
        # logger.info(f'[head]: {pitch:.2f}, yaw: {yaw:.2f}, '
        #             f'roll: {roll:.2f}, distance: {face.distance:.2f}')

    def _draw_landmarks(self, face: Face) -> None:
        if not self.show_landmarks:
            return
        self.visualizer.draw_points(face.landmarks,
                                    color=(0, 255, 255),
                                    size=1)

    def _draw_face_template_model(self, face: Face) -> None:
        if not self.show_template_model:
            return
        self.visualizer.draw_3d_points(face.model3d,
                                       color=(255, 0, 525),
                                       size=1)

    def _display_normalized_image(self, face: Face) -> None:
        if not self.config.demo.display_on_screen:
            return
        if not self.show_normalized_image:
            return
        if self.config.mode == 'MPIIGaze':
            reye = face.reye.normalized_image
            leye = face.leye.normalized_image
            normalized = np.hstack([reye, leye])
        elif self.config.mode in ['MPIIFaceGaze', 'ETH-XGaze']:
            normalized = face.normalized_image
        else:
            raise ValueError
        if self.config.demo.use_camera:
            normalized = normalized[:, ::-1]
        cv2.imshow('normalized', normalized)

    def gaze_area_converter(self, pitch, yaw):
        if pitch > 30:
            if yaw < -40:
                return 0
            elif yaw > 40:
                return 2
            else:
                return 1
        elif pitch < -20:
            if yaw < -40:
                return 6
            elif yaw > 40:
                return 8
            else:
                return 7
        else:
            if yaw < -40:
                return 3
            elif yaw > 40:
                return 5
            else:
                return 4

    def gaze_warning(self):
        gaze_area_counter = [0,0,0,0,0,0,0,0,0]
        for g in self.gaze_area_array:
            gaze_area_counter[g] += 1

        if gaze_area_counter[4] < 10:
            logger.info("Warning! Please Look Ahead")
            sound = AudioSegment.from_mp3('gaze_warning.mp3')
            play(sound)
            self.gaze_area_array = []
            return True
        
        return False

    def daydreaming_warning(self):
        pitch_avg = sum(self.pitch_array) / len(self.pitch_array)
        yaw_avg = sum(self.yaw_array) / len(self.yaw_array)
        pitch_activity, yaw_activity = 0,0

        for p in self.pitch_array:
            diff_p = pow(pitch_avg - p, 2)
            pitch_activity += diff_p

        for y in self.yaw_array:
            diff_y = pow(yaw_avg - y, 2)
            yaw_activity += diff_y

        if pitch_activity < 200 and yaw_activity < 200:
            logger.info("Warning! You Are Daydreaming")
            sound = AudioSegment.from_mp3('daydreaming_warning.mp3')
            play(sound)
            self.pitch_array = []
            self.yaw_array = []

        print(f'[activity] pitch activity:{round(pitch_activity,0)} yaw activity:{round(yaw_activity,0)}')

    def _draw_gaze_vector(self, face: Face) -> None:
        length = self.config.demo.gaze_visualization_length
        if self.config.mode == 'MPIIGaze':
            for key in [FacePartsName.REYE, FacePartsName.LEYE]:
                eye = getattr(face, key.name.lower())
                self.visualizer.draw_3d_line(
                    eye.center, eye.center + length * eye.gaze_vector)
                pitch, yaw = np.rad2deg(eye.vector_to_angle(eye.gaze_vector))
                logger.info(
                    f'[{key.name.lower()}] pitch: {pitch:.2f}, yaw: {yaw:.2f}')
        elif self.config.mode in ['MPIIFaceGaze', 'ETH-XGaze']:
            self.visualizer.draw_3d_line(
                face.center, face.center + length * face.gaze_vector)
            pitch, yaw = np.rad2deg(face.vector_to_angle(face.gaze_vector))
            pitch, yaw = round(pitch, 2), round(yaw, 2)
            self.pitch_array.append(pitch)
            self.yaw_array.append(yaw)
            logger.info(f'[face] pitch: {pitch:.2f}, yaw: {yaw:.2f}')

            #### measure fps ####
            # self.frame_count += 1
            # if self.frame_count % 100 == 0:
            #     now = time.time()
            #     time_spent = now - self.start_time
            #     print(f'time spent for 100 frames: {time_spent}')
            #     self.start_time = now

            if self.config.demo.application:
                gaze_area = self.gaze_area_converter(pitch, yaw)
                self.gaze_area_array.append(gaze_area)
                if len(self.pitch_array) > 100:
                        self.pitch_array = self.pitch_array[1:]
                        self.yaw_array = self.yaw_array[1:]
                if len(self.gaze_area_array) > 50:
                    self.gaze_area_array = self.gaze_area_array[1:]

                    if not self.gaze_warning() and len(self.pitch_array) > 99:
                        self.daydreaming_warning()
                
                    # if self.gaze_warning():
                    #     logger.info("Warning! Please Look Ahead")
                    #     sound = AudioSegment.from_wav('PONPON01.wav')
                    #     play(sound)
                    #     self.gaze_area_array = []
        else:
            raise ValueError
