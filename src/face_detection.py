import numpy as np
from openvino.inference_engine import IECore
import cv2
import logging as log

from model import Model

# CONSTANTS
COLOR_BLUE_BGR = (255, 170, 86)
REQUEST_ID = 0

class FaceDetectionModel(Model):
    '''
    Class for the Face Detection Model.
    '''
    def __init__(self, model_name, device, threshold=0.60):
        Model.__init__(self, model_name, device, 'face_detection_model', threshold)

    def predict(self, image, toggle_face_detect):
        try:
            image_for_prediction = self.preprocess_input(image)
            
            input_dict = {self.input_name: image_for_prediction}
            self.net.start_async(REQUEST_ID, inputs=input_dict)

            if self.net.requests[REQUEST_ID].wait(-1) == 0:
                outputs = self.net.requests[REQUEST_ID].outputs
                outputs = self.preprocess_output(outputs)
                locations, image = self.draw_outputs(outputs, image, toggle_face_detect)
            
            return locations, image
        except Exception as e:
            log.error(f"Error in predict: {e}")

    def draw_outputs(self, coords, image, toggle_face_detect):
        # see create_bounding_boxes function definition for details. 
        # https://github.com/anvillasoto/people-counter-edge-application/blob/master/main.py
        try:
            locations = []
            prob_threshold = float(self.threshold)
            width = int(image.shape[1]) 
            height = int(image.shape[0])
            for coord in coords:
                if coord['confidence'] >= prob_threshold:

                    xmin = int(coord['xmin'] * width)
                    ymin = int(coord['ymin'] * height)
                    xmax = int(coord['xmax'] * width)
                    ymax = int(coord['ymax'] * height)

                    locations.append([xmin,ymin, xmax, ymax])

                    # add box to detected face
                    if toggle_face_detect == 1:
                        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), COLOR_BLUE_BGR, 3)
                    
            return locations, image
        
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
        try:
            bounding_boxes = np.squeeze(outputs[self.output_name])
            
            coords = []
            for box in bounding_boxes:
                if box[0] == -1:
                    return coords
                
                coords.append({
                    "label": box[1],
                    "confidence": box[2],
                    "xmin": box[3],
                    "ymin": box[4],
                    "xmax": box[5],
                    "ymax": box[6]
                })
        
            # per observation, this will be unreachable but we include it just in case
            return coords
        except Exception as e:
            log.error(f"Error in preprocess outputs: {e}")
