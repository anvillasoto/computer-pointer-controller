from openvino.inference_engine import IECore, IENetwork
import cv2
import logging as log

class Model:
    def __init__(self, model_name, device, model_type='', threshold=0.6):
        self.model_weights = model_name + ".bin"
        self.model_structure = model_name + ".xml"
        self.device = device
        self.threshold = threshold

        try:
            self.core = IECore()
            self.model=self.core.read_network(model=self.model_structure, weights=self.model_weights)
        except Exception as e:
            raise ValueError("Could not Initialise the network. Have you enterred the correct model path?")
        
        # gaze estimation has different requirements so it is a special case
        if model_type == 'gaze_estimation_model':
            self.input_name=[i for i in self.model.inputs.keys()]
            self.input_shape=self.model.inputs[self.input_name[1]].shape
        else:
            self.input_name=next(iter(self.model.inputs))
            self.input_shape=self.model.inputs[self.input_name].shape
        self.output_name=next(iter(self.model.outputs))
        self.output_shape=self.model.outputs[self.output_name].shape

    def load_model(self):
        try:
            self.net = self.core.load_network(network=self.model, device_name=self.device, num_requests=1)
        except Exception as e:
            log.error(f"Error in loading model: {e}")
    
    def check_model(self):
        # return model network
        return self.net