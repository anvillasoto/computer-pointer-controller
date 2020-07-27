import numpy as np
import time
from openvino.inference_engine import IECore
import os
import cv2
import argparse
import sys
import logging as log

# models
from face_detection import FaceDetectionModel
from head_pose_estimation import HeadPoseEstimationModel
from facial_landmarks_detection import FacialLandmarksDetectionModel
from gaze_estimation import GazeEstimationModel

# mouse controller
from mouse_controller import MouseController

COLOR_WHITE_BGR = (255, 255, 255)

def main(args):
    # models
    face_detection_model = args.face_detection_model
    head_pose_estimation_model = args.head_pose_estimation_model
    facial_landmarks_detection_model = args.facial_landmarks_detection_model
    gaze_estimation_model = args.gaze_estimation_model

    device=args.device
    video_file=args.video
    threshold=args.threshold
    output_path=args.output_path

    # model load times
    fd_start_model_load_time=time.time()
    fd = FaceDetectionModel(face_detection_model, device, threshold)
    fd.load_model()
    fd_total_model_load_time = time.time() - fd_start_model_load_time

    fld_start_model_load_time=time.time()
    fld = FacialLandmarksDetectionModel(facial_landmarks_detection_model, device)
    fld.load_model()
    fld_total_model_load_time = time.time() - fld_start_model_load_time

    hpe_start_model_load_time=time.time()
    hpe = HeadPoseEstimationModel(head_pose_estimation_model, device)
    hpe.load_model()
    hpe_total_model_load_time = time.time() - hpe_start_model_load_time

    ge_start_model_load_time=time.time()
    ge = GazeEstimationModel(gaze_estimation_model, device)
    ge.load_model()
    ge_total_model_load_time = time.time() - ge_start_model_load_time

    # mouse controller
    mouse_controller = MouseController('medium','fast')

    # Handle the input stream
    # see https://github.com/anvillasoto/people-counter-edge-application/blob/master/main.py
    if video_file == 'CAM':
        input_stream = 0
        single_image_mode = False
    # Checks for input image
    elif video_file.endswith('.jpg') or video_file.endswith('.bmp') :
        single_image_mode = True
        input_stream = video_file
    elif (not video_file.endswith('.jpg')) or (not (video_file.endswith('.bmp'))):
        input_stream = video_file
        assert os.path.isfile(video_file), "Input file does not exist"
    else:
        input_stream = video_file
        log.error("The file is unsupported.please pass a supported file")

    try:
        cap=cv2.VideoCapture(input_stream)
    except Exception as e:
        log.error(f"Something else went wrong with the video file: {e}")
    
    initial_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    initial_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_len = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    out_video = cv2.VideoWriter(os.path.join(output_path, 'output.mp4'), cv2.VideoWriter_fourcc(*'avc1'), fps, (initial_w, initial_h), True)

    counter=0
    start_inference_time=time.time()

    try:
        while cap.isOpened():
            ret, frame=cap.read()
            if not ret:
                break
            counter+=1
            
            # detect face
            face_location, image = fd.predict(frame)
            xmin, ymin, xmax, ymax = face_location[0]
            face_image = image[ymin:ymax, xmin: xmax].copy()
            
            # detect eyes
            eye_locations, eye_images, face_image_drawn = fld.predict(face_image)

            # detect head pose
            head_pose_angles, face_image_drawn = hpe.predict(face_image, face_image_drawn)

            # gaze estimation
            gaze_vector, face_image_drawn = ge.predict(face_image_drawn, eye_images, head_pose_angles, eye_locations)
            image[ymin:ymax, xmin: xmax] = face_image_drawn
            x, y, z = gaze_vector


            # frame message to add gaze vector x, y, and z
            frame_message = "Gaze Coordinates: {:.2f}, {:.2f}, {:.2f}".format(x, y, z)
            image = cv2.putText(image, frame_message, (20, 20), cv2.FONT_HERSHEY_COMPLEX, 1, COLOR_WHITE_BGR, 2)

            out_video.write(image)

            # move mouse after five frames
            if counter % 5 == 0:
                mouse_controller.move(x, y)
            
        total_time=time.time()-start_inference_time
        total_inference_time=round(total_time, 1)
        fps=counter/total_inference_time

        with open(os.path.join(output_path, 'stats.txt'), 'w') as f:
            f.write("Total Inference Time for Four Models: " + str(total_inference_time)+'\n')
            f.write("Frames Per Second for Four Models: " + str(fps)+'\n\n')
            f.write("Model Load Time (Face Detection): " + str(fd_total_model_load_time)+'\n')
            f.write("Model Load Time (Facial Landmark Detection): " + str(fld_total_model_load_time)+'\n')
            f.write("Model Load Time (Head Pose Estimation): " + str(hpe_total_model_load_time)+'\n')
            f.write("Model Load Time (Gaze Estimation): " + str(ge_total_model_load_time)+'\n')

        cap.release()
        cv2.destroyAllWindows()
    except Exception as e:
        print("Could not run Inference: ", e)

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--face_detection_model', required=True,
                        help="Path to face detection model directory")
    parser.add_argument('--head_pose_estimation_model', required=True,
                        help="Path to head pose estimation model directory")
    parser.add_argument('--facial_landmarks_detection_model', required=True,
                        help="Path to facial landmarks detection model directory")
    parser.add_argument('--gaze_estimation_model', required=True,
                        help="Path to gaze estimation model directory")
    parser.add_argument('--device', default='CPU',
                        help="Specify the target device to infer on: "
                             "CPU, GPU, FPGA or MYRIAD is acceptable. Sample "
                             "will look for a suitable plugin for device "
                             "specified (CPU by default)")
    parser.add_argument('--video', default=None,
                        help="Path to image or video file. If you want to use camera, supply 'CAM'")
    parser.add_argument('--output_path', default='/results',
                        help="Output path for drawn video image and statistics.")
    parser.add_argument('--threshold', default=0.60,
                        help="Probability threshold for face detection"
                             "(0.6 by default)")
    
    args=parser.parse_args()

    main(args)