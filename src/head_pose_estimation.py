import numpy as np
from openvino.inference_engine import IECore
import cv2
import logging as log
import math

# CONSTANTS
COLOR_BLUE_BGR = (255, 127, 0)
COLOR_GREEN_BGR = (95, 191, 0)
COLOR_RED_BGR = (86, 86, 255)
REQUEST_ID = 0

class HeadPoseEstimationModel:
    '''
    Class for the Face Detection Model.
    '''
    def __init__(self, model_name, device):
        self.model_weights=model_name+'.bin'
        self.model_structure=model_name+'.xml'
        self.device=device

        try:
            self.core = IECore()
            self.model=self.core.read_network(model=self.model_structure, weights=self.model_weights)
        except Exception as e:
            raise ValueError("Could not Initialise the network. Have you enterred the correct model path?")

        self.input_name=next(iter(self.model.inputs))
        self.input_shape=self.model.inputs[self.input_name].shape
        self.output_name=next(iter(self.model.outputs))
        self.output_shape=self.model.outputs[self.output_name].shape

    def load_model(self):
        try:
            self.net = self.core.load_network(network=self.model, device_name=self.device, num_requests=1)
        except Exception as e:
            log.error(e)

    def predict(self, image, image_to_be_drawn):
        try:
            image_for_prediction = self.preprocess_input(image)
            input_dict = {self.input_name: image_for_prediction}
            self.net.start_async(REQUEST_ID, inputs=input_dict)

            if self.net.requests[REQUEST_ID].wait(-1) == 0:
                outputs = self.net.requests[REQUEST_ID].outputs
                head_pose_angles, image_to_be_drawn = self.draw_outputs(outputs, image, image_to_be_drawn)
            
            return head_pose_angles, image_to_be_drawn
        except Exception as e:
            log.error(f"Error in predict: {e}")

    def check_model(self):
        # i don't need this
        pass

    # from here:
    # https://github.com/hampen2929/pyvino/blob/master/pyvino/model/face_recognition/head_pose_estimation/head_pose_estimation.py
    def build_camera_matrix(self, center_of_face, focal_length):

        cx = int(center_of_face[0])
        cy = int(center_of_face[1])
        camera_matrix = np.zeros((3, 3), dtype='float32')
        camera_matrix[0][0] = focal_length
        camera_matrix[0][2] = cx
        camera_matrix[1][1] = focal_length
        camera_matrix[1][2] = cy
        camera_matrix[2][2] = 1

        return camera_matrix
    
    def draw_outputs(self, coords, image, image_to_be_drawn):
        # see the accepted answer here for details. 
        # https://knowledge.udacity.com/questions/171017 which came from
        # https://github.com/hampen2929/pyvino/blob/master/pyvino/model/face_recognition/head_pose_estimation/head_pose_estimation.py
        try:
            width = image.shape[1] 
            height = image.shape[0]

            yaw = coords['angle_y_fc']
            pitch = coords['angle_p_fc']
            roll = coords['angle_r_fc']

            head_pose_angles = [float(yaw), float(pitch), float(roll)]

            yaw *= np.pi / 180.0
            pitch *= np.pi / 180.0
            roll *= np.pi / 180.0

            center_of_face = (width / 2, height / 2, 0)
            cx = int(center_of_face[0])
            cy = int(center_of_face[1])

            focal_length = 500.0
            scale = 50

            Rx = np.array([[1, 0, 0],
                   [0, math.cos(pitch), -math.sin(pitch)],
                   [0, math.sin(pitch), math.cos(pitch)]])
            Ry = np.array([[math.cos(yaw), 0, -math.sin(yaw)],
                        [0, 1, 0],
                        [math.sin(yaw), 0, math.cos(yaw)]])
            Rz = np.array([[math.cos(roll), -math.sin(roll), 0],
                        [math.sin(roll), math.cos(roll), 0],
                        [0, 0, 1]])
            
            # ref: https://www.learnopencv.com/rotation-matrix-to-euler-angles/
            R = Rz @ Ry @ Rx

            camera_matrix = self.build_camera_matrix(center_of_face, focal_length)

            xaxis = np.array(([1 * scale, 0, 0]), dtype='float32').reshape(3, 1)
            yaxis = np.array(([0, -1 * scale, 0]), dtype='float32').reshape(3, 1)
            zaxis = np.array(([0, 0, -1 * scale]), dtype='float32').reshape(3, 1)
            zaxis1 = np.array(([0, 0, 1 * scale]), dtype='float32').reshape(3, 1)

            o = np.array(([0, 0, 0]), dtype='float32').reshape(3, 1)
            o[2] = camera_matrix[0][0]

            xaxis = np.dot(R, xaxis) + o
            yaxis = np.dot(R, yaxis) + o
            zaxis = np.dot(R, zaxis) + o
            zaxis1 = np.dot(R, zaxis1) + o

            xp2 = (xaxis[0] / xaxis[2] * camera_matrix[0][0]) + cx
            yp2 = (xaxis[1] / xaxis[2] * camera_matrix[1][1]) + cy
            p2 = (int(xp2), int(yp2))
            cv2.line(image_to_be_drawn, (cx, cy), p2, COLOR_RED_BGR, 2)

            xp2 = (yaxis[0] / yaxis[2] * camera_matrix[0][0]) + cx
            yp2 = (yaxis[1] / yaxis[2] * camera_matrix[1][1]) + cy
            p2 = (int(xp2), int(yp2))
            cv2.line(image_to_be_drawn, (cx, cy), p2, (0, 255, 0), 2)

            xp1 = (zaxis1[0] / zaxis1[2] * camera_matrix[0][0]) + cx
            yp1 = (zaxis1[1] / zaxis1[2] * camera_matrix[1][1]) + cy
            p1 = (int(xp1), int(yp1))
            xp2 = (zaxis[0] / zaxis[2] * camera_matrix[0][0]) + cx
            yp2 = (zaxis[1] / zaxis[2] * camera_matrix[1][1]) + cy
            p2 = (int(xp2), int(yp2))

            cv2.line(image_to_be_drawn, p1, p2, (255, 0, 0), 2)
            cv2.circle(image_to_be_drawn, p2, 3, (255, 0, 0), 2)

            return head_pose_angles, image_to_be_drawn
        
        except Exception as e:
            log.error(f"Error in draw_outputs: {e}")

    def preprocess_input(self, image):
        # see my preprocess_image implementation from
        # https://github.com/anvillasoto/people-counter-edge-application/blob/master/main.py
        try:
            image = cv2.resize(image, (self.input_shape[3], self.input_shape[2]))
            image = image.transpose((2, 0, 1))
            image = image.reshape(1, *image.shape)

            return image
        except Exception as e:
            log.error(f"Error in preprocess_input: {e}")

    def preprocess_output(self, outputs):
        # there will be no preprocessing involved in this case
        pass
