#pip install PyQt6
#pip install opencv-python
#pip install ultralytics
#pip install omegaconf
#pip install shapely

from ultralytics import YOLO
from PyQt6 import uic
from PyQt6.QtWidgets import QApplication, QFileDialog, QInputDialog, QLabel, QMainWindow, QWidget, QVBoxLayout
from PyQt6.QtGui import QPixmap
from PyQt6 import QtCore
from PIL import Image, ImageOps
from PIL.ImageQt import ImageQt
import os
import cv2
import glob
from shutil import copy
from threading import Thread
import shutil
import numpy as np
from shapely.geometry import Polygon
from shapely import affinity


class SegmImgs:
    def __init__(self,form):
        self.form = form
        self.form.segm_input_browse.clicked.connect(self.pick_input_path)
        self.form.segm_model_browse.clicked.connect(self.pick_yolo_path)
        self.form.segm_output_path_browse.clicked.connect(self.pick_output_path)
        self.form.segm_start.clicked.connect(self.start_segm_thread)        
        self.model = None
        self.classes = []
        self.SThread = None
        
    def pick_input_path(self):
        dialog = QFileDialog()
        folder_path = dialog.getExistingDirectory(None, "Select Input Folder")
        self.form.segm_input_path.setText(folder_path)
        return folder_path
        
    def pick_yolo_path(self):
        dialog = QFileDialog()
        filter = "yolo models(*.pt)"
        model_path = dialog.getOpenFileName(None, "Select Model","",filter)
        print(model_path[0])
        self.form.segm_model_path.setText(model_path[0])
        if os.path.isfile(model_path[0]):
            self.scanby_yolo(model_path[0])        
        return model_path[0]

    def pick_output_path(self):
        dialog = QFileDialog()
        folder_path = dialog.getExistingDirectory(None, "Select Input Folder")
        self.form.segm_output_path.setText(folder_path)
        return folder_path   

    def scanby_yolo(self,model_path):
        try:
            self.model = YOLO(model_path)
            dict = self.model.names
            self.classes = []
            for i in range(len(dict)):
                self.classes.append(dict[i])
            self.form.Segm_model_list.addItems(self.classes)
        except:
            self.model = None
            self.classes = []
            print("Cannot load this model")

    def start_segm_thread(self):
        if self.model != None or self.classes:
            if self.SThread == None:
                self.SThread = Segm_thread(self.form,self.model,self.classes)
                self.SThread.start()
                self.form.segm_start.setText("Stop removing backgrounds")
            else:
                self.form.segm_start.setText("Start removing backgrounds")
                self.SThread.thread_run = False
                self.SThread.join()    
                self.SThread = None
        else:
            print("Define a yolo model!")

class Segm_thread(Thread):
    def __init__(self, form, model, classes):
        Thread.__init__(self)
        self.input_files = []
        self.form = form
        self.model = model
        self.classes = classes
        self.thread_run = True
        
    def run(self):
        self.load_images()
        if self.input_files:
            self.process_images()
        self.form.segm_start.setText("Start removing backgrounds")

    def load_images(self):
        input_path = self.form.segm_input_path.text()
        for file in glob.iglob(input_path+'/**/*.*', recursive=True):
            if self.thread_run:
                if file.endswith(('.png', '.jpeg', '.jpg', '.PNG', '.JPEG', '.JPG')):
                    file = os.path.abspath(file)
                    self.input_files.append(file)
                    print(file)

    #model = YOLO('yolov8n-seg.pt')
    def process_images(self):
        shutil.rmtree(self.form.segm_output_path.text(), ignore_errors=True)
        if not os.path.exists(self.form.segm_output_path.text()):
            os.makedirs(self.form.segm_output_path.text())        
        count = 0        
        for i in range(len(self.input_files)):
            if self.thread_run:
                results = self.model(self.input_files[i], stream=True, retina_masks = True, conf = self.form.segm_model_threshold.value())
                for result in results:
                    image = cv2.imread(self.input_files[i])
                    h, w, c = image.shape
                    zero_mask = np.zeros_like(image[:, :, 0], dtype=np.uint8)                    
                    masks = result.masks.xy
                    for mask in masks:
                        polygon_coords = mask
                        polygon = Polygon(polygon_coords)
                        cv2.fillPoly(zero_mask, [np.array(polygon_coords, dtype=np.int32)], color=255)
                    zero_mask = cv2.GaussianBlur(zero_mask, (0,0), sigmaX=self.form.segm_smooth_ratio.value(), sigmaY=self.form.segm_smooth_ratio.value(), borderType = cv2.BORDER_DEFAULT)
                    #todo transparent background
                    #overlay = np.zeros([h, w, c], dtype=np.uint8)   
                    #overlay[:] = (self.form.segm_background_B.value(),self.form.segm_background_G.value(),self.form.segm_background_R.value())
                    #alpha = self.form.segm_background_transparency.value()*0.01      
                    #transparent_background = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)
                    #background_mask = np.zeros_like(image[:, :, 0], dtype=np.uint8)
                    #print("cropped_image ",cropped_image.shape,"zero_mask",zero_mask.shape,"transparent_background",transparent_background.shape,"background_mask",background_mask.shape)
                    
                    cropped_image = cv2.bitwise_and(image, image, mask=zero_mask)
                    cropped_image[zero_mask==0] = (self.form.segm_background_B.value(),self.form.segm_background_G.value(),self.form.segm_background_R.value())
                    count +=1
                    cv2.imwrite(os.path.abspath(os.path.join(self.form.segm_output_path.text(),str(count)+'.png')), cropped_image)
                    print("Saving to ", os.path.abspath(os.path.join(self.form.segm_output_path.text(),str(count)+'.png')))



class CollectImgs:
    def __init__(self,form):
        self.form = form
        self.form.collect_input_browse.clicked.connect(self.pick_input_path)
        self.form.collect_model_browse.clicked.connect(self.pick_yolo_path)
        self.form.collect_output_path_browse.clicked.connect(self.pick_output_path)
        self.form.collect_start.clicked.connect(self.start_collect_thread)
        self.model = None
        self.classes = []
        self.CThread = None
       
    def pick_input_path(self):
        dialog = QFileDialog()
        folder_path = dialog.getExistingDirectory(None, "Select Input Folder")
        self.form.collect_input_path.setText(folder_path)
        return folder_path

    def pick_yolo_path(self):
        dialog = QFileDialog()
        filter = "yolo models(*.pt)"
        model_path = dialog.getOpenFileName(None, "Select Model","",filter)
        print(model_path[0])
        self.form.collect_model_path.setText(model_path[0])
        if os.path.isfile(model_path[0]):
            self.scanby_yolo(model_path[0])
        return model_path[0]

    def pick_output_path(self):
        dialog = QFileDialog()
        folder_path = dialog.getExistingDirectory(None, "Select Input Folder")
        self.form.collect_output_path.setText(folder_path)
        return folder_path

    def scanby_yolo(self,model_path):
        try:
            self.model = YOLO(model_path)
            dict = self.model.names
            self.classes = []
            for i in range(len(dict)):
                self.classes.append(dict[i])
            self.form.Collect_model_list.addItems(self.classes)
        except:
            self.model = None
            self.classes = []
            print("Cannot load this model")

    def show_frame_in_display(self,file):
        #pixmap = QPixmap(file) 
        image = Image.open(file)      
        image = ImageOps.contain(image, (640,640))
        qim = ImageQt(image)
        pix = QPixmap.fromImage(qim)
        self.form.status.setPixmap(pix) 
        
        
    def start_collect_thread(self):
        if self.model != None or self.classes:
            if self.CThread == None:
                self.CThread = Collect_thread(self.form,self.model,self.classes)
                self.CThread.start()
                self.form.collect_start.setText("Stop collecting images")
            else:
                self.form.collect_start.setText("Start collecting images")
                self.CThread.thread_run = False
                self.CThread.join()    
                self.CThread = None
        else:
            print("Define a yolo model!")


        
class Collect_thread(Thread):
    def __init__(self, form, model, classes):
        Thread.__init__(self)
        self.input_files = []
        self.form = form
        self.model = model
        self.classes = classes
        self.thread_run = True
            
    def run(self):
        self.load_images()
        if self.input_files:
            self.process_images()
        self.form.collect_start.setText("Start collecting images")

    def load_images(self):
        input_path = self.form.collect_input_path.text()
        for file in glob.iglob(input_path+'/**/*.*', recursive=True):
            if self.thread_run:
                if file.endswith(('.png', '.jpeg', '.jpg', '.PNG', '.JPEG', '.JPG')):
                    width, height = self.get_image_size(file)
                    file = os.path.abspath(file)
                    if width>=self.form.collect_filter_input_width_value.value() and height>=self.form.collect_filter_input_height_value.value():
                        self.input_files.append(file)
                        print(file,"Dimensions: ",width, height)
               
                        #self.show_frame_in_display(file)
    
    def process_images(self):
        shutil.rmtree(self.form.collect_output_path.text(), ignore_errors=True)
        count = 0        
        for i in range(len(self.input_files)):
            if self.thread_run:
                results = self.model(self.input_files[i], stream=True, conf = self.form.Collect_model_threshold.value())
                for r in results:
                    boxes = r.boxes
                    filename = r.path
                    width, height = self.get_image_size(filename)
                    image_area = width * height
                    for box in boxes:
                        dir = os.path.abspath(os.path.join(self.form.collect_output_path.text(),self.classes[int(box.cls)]))
                        if not os.path.exists(dir):
                            os.makedirs(dir)
                        img = cv2.imread(filename)
                        x1, y1, x2, y2 = box.xyxy[0]
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) # convert to int values
                        box_area = (x2-x1)*(y2-y1)
                        if self.form.collect_filter_class_detection_value.value()*0.01*image_area<=box_area:
                            center_coords = (int((x2+x1)/2),int((y2+y1)/2))
                            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
                            if self.form.collect_centered_crops_boolean.isChecked():
                                img = self.fit_to_class_crop(img,int(self.form.collect_output_width_value.value()),int(self.form.collect_output_height_value.value()),center_coords)
                            else:
                                img = self.fit_to_image_crop(img,int(self.form.collect_output_width_value.value()),int(self.form.collect_output_height_value.value()))
                            count +=1
                            cv2.imwrite(os.path.abspath(os.path.join(self.form.collect_output_path.text(),self.classes[int(box.cls)],str(count)+'.png')), img)
                            print(os.path.abspath(os.path.join(self.form.collect_output_path.text(),self.classes[int(box.cls)],str(count)+'.png')))

    def fit_to_class_crop(self,img,max_width,max_height,center_coords):
        height, width, channels = img.shape
        width_scaling_factor = max_width / float(width)
        height_scaling_factor = max_height / float(height)
        if width_scaling_factor >= height_scaling_factor:
            scaling_factor = width_scaling_factor
        else:
            scaling_factor = height_scaling_factor
        if width>=height:
            img = cv2.resize(img, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_CUBIC)
            center_coords = (int(scaling_factor*center_coords[0]),int(scaling_factor*center_coords[1]))
            height, width, channels = img.shape
            x_left_shift = (center_coords[0]-(max_width/2))
            x_right_shift = (center_coords[0]+(max_width/2))
            lx = 0
            if x_left_shift>=0 and x_right_shift<width:
                lx = int(center_coords[0]-max_width/2)
            if x_right_shift>=width:
                lx = width-max_width
            img = img[0:max_height, lx:lx+max_width]
        else:
            img = cv2.resize(img, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_CUBIC)
            center_coords = (int(scaling_factor*center_coords[0]),int(scaling_factor*center_coords[1]))
            height, width, channels = img.shape        
            y_top_shift = (center_coords[1]-(max_height/2))
            y_bottom_shift = (center_coords[1]+(max_height/2))
            uy = 0
            if y_top_shift>=0 and y_bottom_shift<height:
                uy = int(center_coords[1]-max_height/2)
            if y_bottom_shift>=height:
                uy = height-max_height
            img = img[uy:uy+max_height, 0:max_width]
        return img

    def fit_to_image_crop(self,img,max_width,max_height):
        height, width, channels = img.shape
        width_scaling_factor = max_width / float(width)
        height_scaling_factor = max_height / float(height)
        if width_scaling_factor >= height_scaling_factor:
            scaling_factor = width_scaling_factor
        else:
            scaling_factor = height_scaling_factor
        if width>=height:
            img = cv2.resize(img, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_CUBIC)
            height, width, channels = img.shape
            lx = int(width/2-max_width/2)
            img = img[0:max_height, lx:lx+max_width]
        else:
            img = cv2.resize(img, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_CUBIC)
            height, width, channels = img.shape
            uy = int(height/2-max_height/2)
            img = img[uy:uy+max_height, 0:max_width]
        return img


    def get_image_size(self,file_path):
        im = Image.open(file_path)
        width, height = im.size
        return width, height




class MainClass:
    def __init__(self):
        self.Form, self.Window =  uic.loadUiType("DatasetSwiffKnife.ui")
        self.app = QApplication([])
        self.window, self.form = self.Window(), self.Form()
        self.form.setupUi(self.window)
        self.window.show()
        self.CI = CollectImgs(self.form)
        self.SI = SegmImgs(self.form)
        self.app.exec()
        



if __name__ == "__main__":
    MC = MainClass()   