# Object Detection in video

## Project Overview:
This project utilizes YOLO3, an object detection model, to identify objects within images or videos. It counts the identified objects, calculates their speeds, and generates routes for each detected object within the video frames.

### YOLO3 (You Only Look Once):
YOLO is an abbreviation for "You Only Look Once." YOLO is a real-time object detection system that works by dividing an image into a grid and applying bounding boxes and class probabilities to each grid cell. YOLO3, in particular, is an iteration of this model and serves as the backbone for object recognition within this project.

## Installation:
1. Download the following YOLO3 model files:
   - `yolov3.weights`
   - `yolov3.pt`
   - `yolov3.cfg`
   - `yolov3.txt`

 - yolov3.weights: Contains the pre-trained weights of the YOLOv3 model, crucial for object detection.
 - yolov3.pt: Represents the PyTorch format of the YOLOv3 model, facilitating compatibility and usage within PyTorch-based frameworks.
 - yolov3.cfg: Holds the configuration file defining the architecture and parameters of the YOLOv3 neural network.
 - yolov3.txt: Contains the list of class names or labels that the YOLOv3 model is trained to recognize, corresponding to different object categories.


2. Additionally, ensure the installation of Tesseract OCR by downloading and placing `tesseract.exe` at the location: `\Program Files\Tesseract-OCR\tesseract.exe`

3. The project is based on Python and requires PyCharm.

## Explanation of Files:
- `unique_routes.txt`: Records the routes taken by detected objects within the video.
- `detected_objects/`: Contains images of all detected objects captured from various angles within the video frames.
- `video/`: Holds the video files. Replace these files with your own videos for analysis.

## Running the Program:
To initiate the project:
1. Open the `yolomovie.py` file within PyCharm.
2. Run the script.
3. The program will print the trajectories of identified objects within the video.

## Exiting the Program:
To exit the program:
- Press the `ESC` button.



