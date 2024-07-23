# DiffusersDatasetSwissKnife
GUI app for collecting images and backgrounds removing
App can help to collect images from your PC using pretrained yolo model. Model is looking for objects (pretrained yolo classes) on images and tries prepare your images by making object center-cropping with minimum losses.
You can define desired input and output resolution also.

Background removing tab does segmentation (by pretrained yolo classes) on your images and removes all background areas that yolo model cannot segment. Helps to decrease convergence time during training diffusers

Install:
- python for your OS
- install graphics card drivers and Cuda

pip install PyQt6
pip install opencv-python
pip install ultralytics
pip install omegaconf
pip install shapely

Using:
python start.py

