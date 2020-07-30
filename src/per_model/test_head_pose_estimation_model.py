import numpy as np
import time
from openvino.inference_engine import IECore
import os
import cv2
import argparse
import sys
import logging as log
import math
import json

from model import Model

# CONSTANTS
COLOR_BLUE_BGR = (255, 127, 0)
COLOR_GREEN_BGR = (95, 191, 0)
COLOR_RED_BGR = (86, 86, 255)

class TestHeadPoseEstimation(Model):
    '''
    Class for the Test Head Pose Estimation.
    '''

    def __init__(self, model_name, device):
        Model.__init__(self, model_name, device, 'head_pose_estimation_model')
        
    def predict(self, image):
        request_id = 0
        try:
            image_for_prediction = self.preprocess_input(image)
            input_dict = {self.input_name: image_for_prediction}
            self.net.start_async(request_id, inputs=input_dict)

            if self.net.requests[request_id].wait(-1) == 0:
                outputs = self.net.requests[request_id].outputs
                head_pose_angles, image = self.draw_outputs(outputs, image)
            
            return head_pose_angles, image
        except Exception as e:
            log.error(f"Error in predict: {e}")

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
    
    def draw_outputs(self, coords, image):
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
            cv2.line(image, (cx, cy), p2, COLOR_RED_BGR, 2)

            xp2 = (yaxis[0] / yaxis[2] * camera_matrix[0][0]) + cx
            yp2 = (yaxis[1] / yaxis[2] * camera_matrix[1][1]) + cy
            p2 = (int(xp2), int(yp2))
            cv2.line(image, (cx, cy), p2, (0, 255, 0), 2)

            xp1 = (zaxis1[0] / zaxis1[2] * camera_matrix[0][0]) + cx
            yp1 = (zaxis1[1] / zaxis1[2] * camera_matrix[1][1]) + cy
            p1 = (int(xp1), int(yp1))
            xp2 = (zaxis[0] / zaxis[2] * camera_matrix[0][0]) + cx
            yp2 = (zaxis[1] / zaxis[2] * camera_matrix[1][1]) + cy
            p2 = (int(xp2), int(yp2))

            cv2.line(image, p1, p2, (255, 0, 0), 2)
            cv2.circle(image, p2, 3, (255, 0, 0), 2)

            return head_pose_angles, image
        
        except Exception as e:
            log.error(f"Error in draw_outputs: {e}")

    def preprocess_outputs(self, outputs):
        pass

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
    precision = args.precision

    start_model_load_time=time.time()
    tfld = TestHeadPoseEstimation(model, device)
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
    out_video = cv2.VideoWriter(os.path.join(output_path, 'test_head_pose_estimation_model.mp4'), cv2.VideoWriter_fourcc(*'avc1'), fps, (initial_w, initial_h), True)

    counter=0
    start_inference_time=time.time()

    head_pose_angles_list = []

    try:
        while cap.isOpened():
            ret, frame=cap.read()
            if not ret:
                break
            counter+=1
            
            head_pose_angles, image = tfld.predict(frame)
            head_pose_angles_list.append(head_pose_angles)
            
            out_video.write(image)
            
        total_time=time.time()-start_inference_time
        total_inference_time=round(total_time, 1)
        fps=counter/total_inference_time

        with open(os.path.join(output_path, 'stats_' + device + "_" + precision +'.txt'), 'w') as f:
            f.write('Total Inference Time: '+ str(total_inference_time) + '\n')
            f.write('FPS: ' + str(fps) + '\n')
            f.write('Total Model Load Time: ' + str(total_model_load_time) +'\n')

        with open(os.path.join(output_path, 'head_pose_angles_list.txt'), 'w', encoding='utf-8') as fout:
            json.dump(head_pose_angles_list, fout)

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