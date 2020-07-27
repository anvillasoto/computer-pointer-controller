import numpy as np
import time
from openvino.inference_engine import IECore
import os
import cv2
import argparse
import sys
import logging as log
from PIL import Image

# CONSTANTS
COLOR_BLUE_BGR = (255, 170, 86)

class TestFaceDetection:
    '''
    Class for the Face Detection Model.
    '''

    def __init__(self, model_name, device, threshold=0.60):
        self.model_weights=model_name+'.bin'
        self.model_structure=model_name+'.xml'
        self.device=device
        self.threshold=threshold

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
        
    def predict(self, image):
        request_id = 0
        try:
            image_for_prediction = self.preprocess_input(image)
            input_dict = {self.input_name: image_for_prediction}
            self.net.start_async(request_id, inputs=input_dict)

            if self.net.requests[request_id].wait(-1) == 0:
                outputs = self.net.requests[request_id].outputs
                outputs = self.preprocess_outputs(outputs)
                locations, image = self.draw_outputs(outputs, image)
            
            return image
        except Exception as e:
            log.error(f"Error in predict: {e}")
    
    def draw_outputs(self, coords, image):
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
                    cv2.rectangle(image, (xmin, ymin), (xmax, ymax), COLOR_BLUE_BGR, 3)

                    locations.append([xmin,ymin, xmax, ymax])
                    cropped_image = image[ymin:ymax, xmin: xmax]
                    cropped_image_resized = cv2.resize(cropped_image, (240, 370), interpolation=cv2.INTER_AREA)
            return locations, cropped_image_resized
        
        except Exception as e:
            log.error(f"Error in draw_outputs: {e}")

    def preprocess_outputs(self, outputs):
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
    threshold=args.threshold
    output_path=args.output_path

    start_model_load_time=time.time()
    tfd= TestFaceDetection(model, device, threshold)
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
    out_video = cv2.VideoWriter(os.path.join(output_path, 'test_face_detection_model.mp4'), cv2.VideoWriter_fourcc(*'avc1'), fps, (240, 370), True)

    counter=0
    start_inference_time=time.time()

    try:
        while cap.isOpened():
            ret, frame=cap.read()
            if not ret:
                break
            counter+=1
            
            image = tfd.predict(frame)
            out_video.write(image)
            
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
    parser.add_argument('--threshold', default=0.60)
    
    args=parser.parse_args()

    main(args)