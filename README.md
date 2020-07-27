# Computer Pointer Controller

Using multiple object detection and recognition models from Intel Pretrained Models and Intel hardware, we create an application that controls mouse pointer aided only by gaze.

## Project Set Up and Installation

### Prerequisites

For this project, the author runs the system in Windows 10 with Windows Subsystem for Linux running Ubuntu 18.04. Intel Distribution of OpenVINO Toolkit Version 2020.4 for Linux and Python 3.7 are also installed. The system is equipped with Intel(R) Core (TM) i7-7600U CPU @2.80GHz (4 CPUs), ~2.9GHz, Intel(R) HD Graphics 620 and 12 GB memory. External peripherals include an integrated camera but for the most part of this project, the author uses a demo footage which can be found in bin/demo.mp4.

### Setting up Environment

The author starts by setting up virtual environment by executing the following commands in the main level working directory:

```
source /opt/intel/openvino/bin/setupvars.sh -pyver 3.7
sudo apt install x11-xserver-utils
sudo pip3 install --upgrade --force-reinstall python3-xlib
sudo apt-get install python3-tk python3-dev
```

Since WSL does not run Linux Desktop Environment (GNOME, some version of KDE, etc.) and display must be accessed by the application for mouse pointer to be controlled autonomously, an extra step shall be made by installing [Xming X Server for Windows](https://sourceforge.net/projects/xming/).

After installing the software, we export DISPLAY environment variable in the session by executing the following command:

```
export DISPLAY=localhost:0.0
```

Finally, we create a separate Python environment for this project called __computer-pointer-env__ which was setup by executing the following in the same working directory:

```
pip3 install -r requirements.txt
virtualenv computer-pointer-env
source computer-pointer-env/bin/activate
```

### Directory Structure

Aside from the environment mentioned before, we have the following directories:

1. __bin__ - contains the demo footage and the output directory for the final video and stats
2. __models__ - contains the models required to run the application
3. __src__ - contains benchmarking scripts as well as primary classes and __main.py__ as the official script for the project
    1. __per_model__ - contains scripts for benchmarking each model in their separate Python scripts
        1. __outputs__ - stores the output for benchmarking each model

__requirements.txt__ in the other hand contains dependencies necessary to run the project.

### Installing Required Models

Assuming that the complete package for Linux is installed, the toolkit is equipped with model downloader. We install the following models:

1. __face-detection-adas-binary-0001__ for face detection
2. __landmarks-regression-retail-0009__ for landmark detection particularly extracting eye coordinates
3. __head-pose-estimation-adas-0001__ for detecting head pose using Tait-Bryan angles
4. __gaze-estimation-adas-0002__ for extracting gaze direction vector from valid inputs

We install them by executing the following in the main level directory

```
/opt/intel/openvino/deployment_tools/open_model_zoo/tools/downloader/downloader.py --name face-detection-adas-binary-0001 -o models/

/opt/intel/openvino/deployment_tools/open_model_zoo/tools/downloader/downloader.py --name head-pose-estimation-adas-0001 -o models/

/opt/intel/openvino/deployment_tools/open_model_zoo/tools/downloader/downloader.py --name landmarks-regression-retail-0009 -o models/

/opt/intel/openvino/deployment_tools/open_model_zoo/tools/downloader/downloader.py --name gaze-estimation-adas-0002 -o models/
```

## Demo

![01_output](images/01_output.gif)

To run the application, execute the following in the __src__ directory:

```
cd src
Main File
python3 main.py --face_detection_model "../models/intel/face-detection-adas-binary-0001/FP32-INT1/          face-detection-adas-binary-0001" --head_pose_estimation_model "../models/intel/head-pose-estimation-adas-0001/FP32/head-pose-estimation-adas-0001" --facial_landmarks_detection_model "../models/intel/landmarks-regression-retail-0009/FP32/landmarks-regression-retail-0009" --gaze_estimation_model "../models/intel/gaze-estimation-adas-0002/FP32/gaze-estimation-adas-0002" --device CPU --video "../bin/demo.mp4" --output_path "../bin/output/" --threshold 0.50
```

To see the mouse pointer in action particularly in Windows running Ubuntu WSL, run __Xming Server__ or double click from the system tray if available.

## Documentation

Running the application is pretty straightforward as long as you follow the instructions from the project setup above. In this section, the author would like to interest you to the inputs arguments as shown in the script section of the demo:

1. __--face_detection_model__ - path to face detection model directory
2. __--head_pose_estimation_model__ - path to head pose estimation model directory
3. __--facial_landmarks_detection_model__ - path to facial landmarks detection model directory
4. __--gaze_estimation_model__ - path to gaze estimation model directory
5. __--device__- specify the target device to infer on CPU, GPU, FPGA or MYRIAD is acceptable. Sample will look for a suitable plugin for device specified (CPU by default
6. __--video__ - path to image or video file. If you want to use camera, supply 'CAM'
7. __--output_path__ - output path for drawn video image and statistics
8. __--threshold__ - probability threshold for face detection (0.6 by default)

## Benchmarks

For the benchmarking, the author created separate scripts for each models supported which can be found in __src/per_model__ directory. Basically it is the stripped down version on how to run main.py file as explained in the previous section which can be run independently. The order of which they should run can be seen in scripts that follow:

```
cd src/per_model

# Face Detection Model:
python3 test_face_detection_model.py --model "../../models/intel/face-detection-adas-binary-0001/FP32-INT1/face-detection-adas-binary-0001" --video "../../bin/demo.mp4" --output_path outputs/test_face_detection_model/ --threshold 0.5

# Facial Landmark Detection Model
python3 test_facial_landmarks_detection_model.py --model "../../models/intel/landmarks-regression-retail-0009/FP32/landmarks-regression-retail-0009" --video "outputs/test_face_detection_model/test_face_detection_model.mp4" --output_path "outputs/test_facial_landmarks_detection_model/"

# Head Pose Estimation Model
python3 test_head_pose_estimation_model.py --model "../../models/intel/head-pose-estimation-adas-0001/FP32/head-pose-estimation-adas-0001" --video "outputs/test_face_detection_model/test_face_detection_model.mp4" --output_path "outputs/test_head_pose_estimation_model/"

# Gaze Estimation Model
python3 test_gaze_estimation_model.py --model "../../models/intel/gaze-estimation-adas-0002/FP32/gaze-estimation-adas-0002" --video "../../bin/demo.mp4" --output_path "outputs/test_gaze_estimation_model/"
```

### Face Detection Model Benchmark

For the face detection model, we use the __test_face_detection_model.py__ with inputs like model directory, video feed, output path and probability threshold. The author specifically cropped each of the faces per frame from the video feed, resized each of them in 240 x 370 resolution and saved in a video file to be used in the following two benchmarks. The resulting video can be seen below:

![02_test_face_detection_model](images/02_test_face_detection_model.gif)

### Facial Landmark Detection Model Benchmark

For the facial landmark detection, we use the __test_facial_landmarks_detection_model.py__ with inputs the same as what face detection model excluding the threshold. The video feed that will be used in this case is the output of the face detection model. The output of this script is the same video output with left and right eye bounding boxes drawn per frame like what you see below:

![test_facial_landmarks_detection_model](images/03_test_facial_landmarks_detection_model.gif)

Another output is a saved text file for left and right eye images extracted per frame, put in the dictionary and appended into a list that will be used as part of the gaze estimation inputs. This text file can be found on [__src/per_model/outputs/test_facial_landmarks_detection_model/eye_images_list.txt__](src/per_model/outputs/test_facial_landmarks_detection_model/eye_images_list.txt)

### Head Pose Estimation Model Benchmark

For the head pose estimation model benchmark, we use the __test_head_pose_estimation_model.py__ with arguments quite the same as the facial landmark detection model .py file since head pose estimation model needs a cropped head image to perform prediction. The output is a video file with Euler angles drawn in the middle of each frame signifying head pose direction in three dimensions which you can see below:

![04_test_head_pose_estimation_model](images/04_test_head_pose_estimation_model.gif)

Another output is the list of Tait-Bryan angles or the output of each prediction per frame saved in the text file which is also a required input for gaze estimation model benchmark. You can find the said file [here](src/per_model/outputs/test_head_pose_estimation_model).

### Gaze Estimation Model Benchmark

Since gaze estimation needs actual cropped images of left and right eyes as well as the Tait-Bryan angles per inference, the author saved these in their respective text files as mentioned above. This is to maintain the independent nature of each scripts, in that, despite extra time to read each file sequentially, it can be ran separately and get meaningful results like the video feeds presented above. 

The process is to load the __eye_images_list.txt__ and __head_pose_angles_list.txt__ in-memory, iterate through each of them as if they are parts of separate frame including the frame from the original demo video, run inference to these inputs and draw line at the center of each frame, signifying to which position the person is looking per frame. The author also displays these gaze estimation coordinates in the upper left corner of each frame.

![05_test_gaze_estimation_model](images/05_test_gaze_estimation_model.gif)


## Results

Due to limitation of hardware as discussed in the Prerequisites subsection of Project Set Up and Installation, the author can only report the performance of each model reported by the outputs of the scripts executed from above in CPU and FP32 precisions.

The result is divided in statistics per model. You can find the statistical output of each script run from four models as well as the main.py file in the following files:

1. [__src/per_model/outputs/test_face_detection_model/stats.txt__](src/per_model/outputs/test_face_detection_model/stats.txt)
2. [__src/per_model/outputs/test_facial_landmarks_detection_model/stats.txt__](src/per_model/outputs/test_facial_landmarks_detection_model/stats.txt)
3. [__src/per_model/outputs/test_gaze_estimation_model/stats.txt__](src/per_model/outputs/test_gaze_estimation_model/stats.txt)
4. [__src/per_model/outputs/test_head_pose_estimation_model/stats.txt__](src/per_model/outputs/test_head_pose_estimation_model/stats.txt)

### Results per Model

Interestingly, results per model on total model load time is significantly fast. Frames per second in the other hand is surprisingly high for facial landmark detection and head pose detection. For face detection and gaze estimation, they are roughly in the 30 FPS territory which is unsurprising because for the former, it not only needs to account for the bounding box for a detected face but also the possibility of the existence of more than one face in the frame not to mention the orientation of the face detected and for the latter, which needs to account for both eye images and head pose which are quite computationally expensive. 

Specifically, the following are the result per model.

#### Face Detection Model:

Total Inference Time: 17.8
FPS: 33.42696629213483
Total Model Load Time: 0.5270497798919678

#### Facial Landmark Detection Model:

Total Inference Time: 2.7
FPS: 220.37037037037035
Total Model Load Time: 0.0960690975189209

#### Head Pose Estimation Model:

Total Inference Time: 3.0
FPS: 198.33333333333334
Total Model Load Time: 0.14165019989013672

#### Gaze Estimation Model:

Total Inference Time: 30.2
FPS: 19.70198675496689
Total Model Load Time: 0.11391782760620117

## Stand Out Suggestions

Aside from feeding the application with a video file, we can also include a video feed from camera using CAM as input to video argument. We can run this by executing the following command:

```
python3 main.py --face_detection_model "../models/intel/face-detection-adas-binary-0001/FP32-INT1/face-detection-adas-binary-0001" --head_pose_estimation_model "../models/intel/head-pose-estimation-adas-0001/FP32/head-pose-estimation-adas-0001" --facial_landmarks_detection_model "../models/intel/landmarks-regression-retail-0009/FP32/landmarks-regression-retail-0009" --gaze_estimation_model "../models/intel/gaze-estimation-adas-0002/FP32/gaze-estimation-adas-0002" --device CPU --video "CAM" --output_path "../bin/output/" --threshold 0.50
```

### Async Inference

All of the implementations from benchmarking to running the main application uses asynchronous inference on a per-frame basis. The benefit of this compare to its synchronous/blocking counterpart is its non-blocking nature in that it can run on background without blocking the whole application, spitting out result right where it is finished in inference. Although the author did not take advantage the batch processing per frame, it is next in his pipeline for system improvement. Asynchronous inference for multiple pipeline is presented in the following output:

1. [__bin/output/stats.txt__](bin/output/stats.txt) - includes the overall total inference time and frames per second for four models combined.

#### Overall Result:

This is incredibly surprising because combining four models not to mention image manipulation per frame is surprisingly effective in CPU. For a 19-second video, frames per second boasted a total of 3 frames per second with a total inference time of 175.8 seconds. With help from multiple accelerators available, we can boost this up more not to mention take advantage of batch processing in IGPU environments.

Total Inference Time for Four Models: 175.8
Frames Per Second for Four Models: 3.38452787258248

### Edge Cases

Obviously, this application is not without its limitations. Firstly, its sequential nature has multiple points of failure especially when former models before gaze estimation encounter problems (i.e. face detection fails because there is no face detected in a single frame). We solve that by ignoring problematic frames and going to the next frame, essentially preventing the application to fail once it encounters that error. 

Another is face detection, like the author said earlier can detect multiple faces in the frame. To remedy this, the model only returns the first one in the result and that will be the basis of succeeding models that depend on it.

Finally, the reason behind the separation of benchmark per model is to simulate how each of them perform on independent environments. Of course, we can just time the model load and inference times from the main file but it will be easy for other users to use these models independently with a completed pipeline from these scripts. 