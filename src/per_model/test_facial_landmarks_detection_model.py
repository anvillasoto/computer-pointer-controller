import numpy as np
import time
from openvino.inference_engine import IECore
import os
import cv2
import argparse
import sys
import logging as log
import json

from model import Model

# CONSTANTS
COLOR_GREEN_BGR = (95, 191, 0)

class TestFacialLandmarksDetection(Model):
    '''
    Class for the Test Facial Landmark Detection.
    '''

    def __init__(self, model_name, device):
        Model.__init__(self, model_name, device, 'facial_landmarks_detection_model')
        
    def predict(self, image):
        request_id = 0
        try:
            image_for_prediction = self.preprocess_input(image)
            input_dict = {self.input_name: image_for_prediction}
            self.net.start_async(request_id, inputs=input_dict)

            if self.net.requests[request_id].wait(-1) == 0:
                outputs = self.net.requests[request_id].outputs[self.output_name]
                coords = self.preprocess_outputs(outputs)
                eye_images, image = self.draw_outputs(coords, image)
            
            return eye_images, image
        except Exception as e:
            log.error(f"Error in predict: {e}")
    
    def draw_outputs(self, coords, image):
        # see the accepted answer here for details. 
        # https://knowledge.udacity.com/questions/245775
        try:
            width = image.shape[1] 
            height = image.shape[0]
            
            # "left_eye_x": left_eye_x,
            # "left_eye_y": left_eye_y,
            # "right_eye_x": right_eye_x,
            # "right_eye_y": right_eye_y
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

            left_eye_image = image[left_eye_ymin:left_eye_ymax, left_eye_xmin: left_eye_xmax].copy()
            left_eye_image = self.preprocess_input(left_eye_image, type='eye')
            
            right_eye_image = image[right_eye_ymin: right_eye_ymax, right_eye_xmin: right_eye_xmax].copy()
            right_eye_image = self.preprocess_input(right_eye_image, type='eye')

            # draw rectangle for each eye
            cv2.rectangle(image, (left_eye_xmin, left_eye_ymin), (left_eye_xmax, left_eye_ymax), COLOR_GREEN_BGR, 2)
            cv2.rectangle(image, (right_eye_xmin, right_eye_ymin), (right_eye_xmax, right_eye_ymax), COLOR_GREEN_BGR, 2)

            eye_images = {"left_eye_image": left_eye_image.tolist(), "right_eye_image": right_eye_image.tolist()}

            return eye_images, image
        
        except Exception as e:
            log.error(f"Error in draw_outputs: {e}")

    def preprocess_outputs(self, outputs):
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

    def preprocess_input(self, image, type='landmark'):
        # see my preprocess_image implementation from
        # https://github.com/anvillasoto/people-counter-edge-application/blob/master/main.py
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

def main(args):
    model=args.model
    device=args.device
    video_file=args.video
    output_path=args.output_path
    precision=args.precision

    start_model_load_time=time.time()
    tfld = TestFacialLandmarksDetection(model, device)
    tfld.load_model()
    total_model_load_time = time.time() - start_model_load_time

    try:
        cap=cv2.VideoCapture(video_file)
    except FileNotFoundError:
        print("Cannot locate video file: "+ video_file)
    except Exception as e:
        print("Something else went wrong with the video file: ", e)
    
    initial_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    initial_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_len = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    out_video = cv2.VideoWriter(os.path.join(output_path, 'test_facial_landmarks_detection_model.mp4'), cv2.VideoWriter_fourcc(*'avc1'), fps, (initial_w, initial_h), True)

    counter=0
    start_inference_time=time.time()

    eye_images_list = []

    try:
        while cap.isOpened():
            ret, frame=cap.read()
            if not ret:
                break

            counter+=1
            
            eye_images, image= tfld.predict(frame)
            
            out_video.write(image)

            eye_images_list.append(eye_images)
            
        total_time=time.time()-start_inference_time
        total_inference_time=round(total_time, 1)
        fps=counter/total_inference_time

        with open(os.path.join(output_path, 'stats_' + device + "_" + precision +'.txt'), 'w') as f:
            f.write('Total Inference Time: '+ str(total_inference_time) + '\n')
            f.write('FPS: ' + str(fps) + '\n')
            f.write('Total Model Load Time: ' + str(total_model_load_time) +'\n')

        with open(os.path.join(output_path, 'eye_images_list.txt'), 'w', encoding='utf-8') as fout:
            json.dump(eye_images_list, fout)

        cap.release()
        cv2.destroyAllWindows()
    except Exception as e:
        print("Could not run Inference: ", e)

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--device', default='CPU')
    parser.add_argument('--video', default=None)
    parser.add_argument('--output_path', default='/results')
    parser.add_argument('--precision', default='FP32')
    
    args=parser.parse_args()

    main(args)