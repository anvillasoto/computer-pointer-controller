import numpy as np
from openvino.inference_engine import IECore
import cv2
import logging as log

from model import Model

# CONSTANTS
COLOR_GREEN_BGR = (95, 191, 0)
REQUEST_ID = 0

class FacialLandmarksDetectionModel(Model):
    '''
    Class for the Face Detection Model.
    '''
    def __init__(self, model_name, device):
        Model.__init__(self, model_name, device, 'facial_landmarks_detection_model')

    def predict(self, image, toggle_eye_detection):
        try:
            image_for_prediction = self.preprocess_input(image)
            input_dict = {self.input_name: image_for_prediction}
            self.net.start_async(REQUEST_ID, inputs=input_dict)

            if self.net.requests[REQUEST_ID].wait(-1) == 0:
                outputs = self.net.requests[REQUEST_ID].outputs[self.output_name]
                coords = self.preprocess_output(outputs)
                eye_locations, eye_images, image = self.draw_outputs(coords, image, toggle_eye_detection)
            
            return eye_locations, eye_images, image
        except Exception as e:
            log.error(f"Error in predict: {e}")

    def draw_outputs(self, coords, image, toggle_eye_detection):
        # see the accepted answer here for details. 
        # https://knowledge.udacity.com/questions/245775
        try:
            width = image.shape[1] 
            height = image.shape[0]
            
            left_eye_x = coords['left_eye_x'] * width
            left_eye_y = coords['left_eye_y'] * height
            right_eye_x = coords['right_eye_x'] * width
            right_eye_y = coords['right_eye_y'] * height

            left_eye_xmin = int(left_eye_x - 20)
            left_eye_ymin = int(left_eye_y - 20)
            right_eye_xmin = int(right_eye_x - 20)
            right_eye_ymin = int(right_eye_y - 20)

            left_eye_xmax = int(left_eye_x + 20)
            left_eye_ymax = int(left_eye_y + 20)
            right_eye_xmax = int(right_eye_x + 20)
            right_eye_ymax = int(right_eye_y + 20)

            eye_locations = {
                "left_eye_location": [left_eye_x, left_eye_y], 
                "right_eye_location": [right_eye_x, right_eye_y]}

            left_eye_image = image[left_eye_ymin:left_eye_ymax, left_eye_xmin: left_eye_xmax].copy()
            left_eye_image = self.preprocess_input(left_eye_image, type='eye')
            
            right_eye_image = image[right_eye_ymin: right_eye_ymax, right_eye_xmin: right_eye_xmax].copy()
            right_eye_image = self.preprocess_input(right_eye_image, type='eye')

            # draw rectangle for each eye based on toggle
            if toggle_eye_detection == 1:
                cv2.rectangle(image, (left_eye_xmin, left_eye_ymin), (left_eye_xmax, left_eye_ymax), COLOR_GREEN_BGR, 2)
                cv2.rectangle(image, (right_eye_xmin, right_eye_ymin), (right_eye_xmax, right_eye_ymax), COLOR_GREEN_BGR, 2)

            eye_images = {"left_eye_image": left_eye_image, "right_eye_image": right_eye_image}

            return eye_locations, eye_images, image
        
        except Exception as e:
            log.error(f"Error in draw_outputs: {e}")

    def preprocess_input(self, image, type='landmark'):
        # see my preprocess_image implementation from
        # https://github.com/anvillasoto/people-counter-edge-application/blob/master/main.py
        
        # due to different requirements for both landmark and eye (for gaze estimation) models,
        # resize may have different implementation as defined by the conditional below.
        try:
            if type == 'landmark':
                image = cv2.resize(image, (self.input_shape[3], self.input_shape[2]))
            elif type == 'eye':
                image = cv2.resize(image, (60, 60), interpolation=cv2.INTER_AREA)
            image = image.transpose((2, 0, 1))
            image = image.reshape(1, *image.shape)
            return image
        except Exception as e:
            log.error(f"Error in preprocess_input: {e}")

    def preprocess_output(self, outputs):
        # landmark detection outputs
        # https://knowledge.udacity.com/questions/245775
        try:

            coords = outputs[0]
        
            left_eye_x, left_eye_y = coords[0][0], coords[1][0]
            right_eye_x, right_eye_y = coords[2][0], coords[3][0]

            return {
                "left_eye_x": left_eye_x,
                "left_eye_y": left_eye_y,
                "right_eye_x": right_eye_x,
                "right_eye_y": right_eye_y
            }
        except Exception as e:
            log.error(f"Error in preprocess outputs: {e}")
