import numpy as np
from openvino.inference_engine import IECore
import cv2
import logging as log

# CONSTANTS
COLOR_VIOLET_BGR = (255, 0, 255)
REQUEST_ID = 0

class GazeEstimationModel:
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

        self.input_name=[i for i in self.model.inputs.keys()]
        self.input_shape=self.model.inputs[self.input_name[1]].shape
        self.output_name=next(iter(self.model.outputs))
        self.output_shape=self.model.outputs[self.output_name].shape

    def load_model(self):
        try:
            self.net = self.core.load_network(network=self.model, device_name=self.device, num_requests=1)
        except Exception as e:
            log.error(e)

    def predict(self, image, eye_images, head_pose_angles, eye_locations):
        try:
            image_for_prediction = self.preprocess_input(image)
            left_eye_image = np.array(eye_images['left_eye_image'])
            right_eye_image = np.array(eye_images['right_eye_image'])
            
            input_dict = {"left_eye_image": left_eye_image, 'right_eye_image': right_eye_image, 'head_pose_angles': head_pose_angles}
            self.net.start_async(REQUEST_ID, inputs=input_dict)

            if self.net.requests[REQUEST_ID].wait(-1) == 0:
                outputs = self.net.requests[REQUEST_ID].outputs
                outputs = self.preprocess_outputs(outputs)
                gaze_vector, image = self.draw_outputs(outputs, image, eye_locations)
            
            return gaze_vector, image
        except Exception as e:
            log.error(f"Error in predict: {e}")

    def check_model(self):
        # i don't need this
        pass

    def draw_outputs(self, gaze_vector, image, eye_locations):
        # see create_bounding_boxes function definition for details. 
        # drawing one line from center for benchmarking purposes
        # from https://knowledge.udacity.com/questions/257811
        try:
            x = gaze_vector[0]
            y = gaze_vector[1]
            z = gaze_vector[2]

            left_eye_location = eye_locations['left_eye_location']
            right_eye_location = eye_locations['right_eye_location']

            left_eye_x, left_eye_y = left_eye_location
            right_eye_x, right_eye_y = right_eye_location

            center_of_left_eye = (int(left_eye_x), int(left_eye_y))
            center_of_right_eye = (int(right_eye_x), int(right_eye_y))
            
            image = cv2.line(image, center_of_left_eye, (int(center_of_left_eye[0] + x * 200), int(center_of_left_eye[1] - y * 200)), COLOR_VIOLET_BGR, 3)
            image = cv2.line(image, center_of_right_eye, (int(center_of_right_eye[0] + x * 200), int(center_of_right_eye[1] - y * 200)), COLOR_VIOLET_BGR, 3)
            
            return gaze_vector, image
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

    def preprocess_outputs(self, outputs):
        return outputs['gaze_vector'][0]
