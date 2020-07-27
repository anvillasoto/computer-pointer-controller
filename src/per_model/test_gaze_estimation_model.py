import numpy as np
import time
from openvino.inference_engine import IECore
import os
import cv2
import argparse
import sys
import logging as log
import ast

# CONSTANTS
COLOR_VIOLET_BGR = (255, 0, 255)

class TestGazeEstimation:
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
        
    def predict(self, image, eye_images, head_pose_angles):
        request_id = 0
        try:
            image_for_prediction = self.preprocess_input(image)
            left_eye_image = np.array(eye_images['left_eye_image'])
            right_eye_image = np.array(eye_images['right_eye_image'])
            input_dict = {"left_eye_image": left_eye_image, 'right_eye_image': right_eye_image, 'head_pose_angles': head_pose_angles}
            self.net.start_async(request_id, inputs=input_dict)

            if self.net.requests[request_id].wait(-1) == 0:
                outputs = self.net.requests[request_id].outputs
                outputs = self.preprocess_outputs(outputs)
                gaze_vector, image = self.draw_outputs(outputs, image)
            
            return gaze_vector, image
        except Exception as e:
            log.error(f"Error in predict: {e}")
    
    def draw_outputs(self, gaze_vector, image):
        # see create_bounding_boxes function definition for details. 
        # drawing one line from center for benchmarking purposes
        # from https://knowledge.udacity.com/questions/257811
        try:
            x = gaze_vector[0]
            y = gaze_vector[1]
            z = gaze_vector[2]

            center_of_face = (int(image.shape[1] / 2), int(image.shape[0] / 2))
            
            image = cv2.line(image, center_of_face, (int(center_of_face[0] + x * 500), int(center_of_face[1] - y * 500)), COLOR_VIOLET_BGR, 5)
            
            frame_message = "Gaze Coordinates: {:.2f}, {:.2f}, {:.2f}".format(x, y, z)
            image = cv2.putText(image, frame_message, (20, 20), cv2.FONT_HERSHEY_COMPLEX, 1, COLOR_VIOLET_BGR, 1)
            
            return gaze_vector, image
        except Exception as e:
            log.error(f"Error in draw_outputs: {e}")

    def preprocess_outputs(self, outputs):
        return outputs['gaze_vector'][0]

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

def main(args):
    model=args.model
    device=args.device
    video_file=args.video
    output_path=args.output_path

    start_model_load_time=time.time()
    tfd= TestGazeEstimation(model, device)
    tfd.load_model()
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
    out_video = cv2.VideoWriter(os.path.join(output_path, 'test_gaze_estimation_model.mp4'), cv2.VideoWriter_fourcc(*'avc1'), fps, (initial_w, initial_h), True)

    # it will take time to load this so be patient

    eye_images_list = []
    with open('outputs/test_facial_landmarks_detection_model/eye_images_list.txt', 'r') as f:
        eye_images_list = ast.literal_eval(f.read())

    head_pose_angles_list = []
    with open('outputs/test_head_pose_estimation_model/head_pose_angles_list.txt', 'r') as f:
        head_pose_angles_list = ast.literal_eval(f.read())

    counter=0
    start_inference_time=time.time()

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            gaze_vector, image = tfd.predict(frame, eye_images_list[counter], head_pose_angles_list[counter])
            out_video.write(image)

            counter+=1
            
        total_time=time.time()-start_inference_time
        total_inference_time=round(total_time, 1)
        fps=counter/total_inference_time

        with open(os.path.join(output_path, 'stats.txt'), 'w') as f:
            f.write(str(total_inference_time)+'\n')
            f.write(str(fps)+'\n')
            f.write(str(total_model_load_time)+'\n')

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
    
    args=parser.parse_args()

    main(args)