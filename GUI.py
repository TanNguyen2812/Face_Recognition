# from MTCNN import MTCNN
import os
import sys
import time

import cv2
import numpy as np
import pandas as pd
import tensorrt
import torch
import torchvision.transforms.functional as F
from numpy.linalg import norm
from PIL import Image
from PyQt5 import QtCore, uic
from PyQt5.QtCore import QFile, Qt, QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (QApplication, QLabel, QMainWindow, QRadioButton,
                             QVBoxLayout, QWidget,QComboBox)


from face_regconition_model import base_transform
from Models.IResnet100_TRT import Iresnet100
from Models.Meta_FAS_TRT import Meta_FAS
from Models.RetinaFace import Retinaface_trt
from read_data import read_data


FAS_LABEL = ["FAKE","REAL"]

def YUV_Mode(img):
    img_yuv = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)

    # equalize the histogram of the Y channel
    img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])

    # convert the YUV image back to RGB format
    img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB)
    return img_output

def YCrBC_Mode(img):
    #### YCrBC
    ycrcb_img = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)

    # equalize the histogram of the Y channel
    ycrcb_img[:, :, 0] = cv2.equalizeHist(ycrcb_img[:, :, 0])

    # convert back to RGB color-space from YCrCb
    equalized_img = cv2.cvtColor(ycrcb_img, cv2.COLOR_YCrCb2RGB)
    return equalized_img

#### Histogram
def Hist_Eq(img):
    for i in range(3):
        img[:,:,i] = cv2.equalizeHist(img[:,:,i])
    return img

def CLAHE_Mode(img,grid=100):

    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

    lab_planes = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.0,tileGridSize=(grid,grid))

    lab_planes[0] = clahe.apply(lab_planes[0])

    lab = cv2.merge(lab_planes)

    img = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    return img


OPTIONS = {'Historgram Eq':Hist_Eq,
           'CLAHE':CLAHE_Mode,
           'YCrBC':YCrBC_Mode,
           'YUV':YUV_Mode
           }


def crop_face(image, output_dir, image_name):
    # Get the bounding box coordinates and dimensions
    face = image
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    filename = os.path.join(output_dir, f"{image_name}.jpg")
    cv2.imwrite(filename, cv2.cvtColor(face,cv2.COLOR_RGB2BGR))
    

class CameraThread(QThread):
    image_data = pyqtSignal(list)
    # image_data = pyqtSignal(np.ndarray)

    def __init__(self, face_detection_model:Retinaface_trt, face_recognition_model:Iresnet100,FAS_model:Meta_FAS):
        super().__init__()
        self.capture = None
        self.save_file = False
        self.save_dir = ''
        self.image_name = 1
        self.face_detection_model = face_detection_model
        self.face_recognition_model = face_recognition_model
        self.FAS_model = FAS_model
        self.mode = 'collect'
        self.stop_thread = False
        self.data = None
        self.mode = None


    def run(self):
        self.face_detection_model.warmup('/images/image1.jpg')
        self.capture = cv2.VideoCapture(0)
        # self.capture.set(cv2.CAP_PROP_EXPOSURE, -3)
        while True:
            ret, frame = self.capture.read()
            if ret:
                frame = cv2.flip(frame,1)
                frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
                # Run face detection model on frame
                # Crop Image is RGB
                # times = time.perf_counter()
                crop, bboxes = self.face_detection_model.Detect_n_Align(frame)
                if bboxes is not None:
                    for bbox in bboxes:
                        x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                        if self.mode != 'None':
                            crop = OPTIONS[self.mode](crop)
                        pred_score,image_tex = self.FAS_model.classify(crop)
                        image_tex = np.transpose(image_tex*255,(1,2,0)).copy()
                        image_tex = np.uint8(image_tex)
                        label = 1 if pred_score >=0.21 else 0
                        # print(pred_score)
                        cv2.putText(frame, FAS_LABEL[label], (x1, y1 - 50),
                                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                        if label==0:continue
            
                        if self.save_file and self.mode == 'collect':
                            path = os.path.join('data', self.save_dir)
                            crop_face(crop, output_dir=path, image_name=str(self.image_name))
                            emb_vec = self.extract_featture(crop)
                            pd.to_pickle(emb_vec, os.path.join(path, str(self.image_name)+'.pkl'))
                            self.save_file = False
                            self.image_name += 1 # mode

                        if self.mode == 'run':
                            emb_vec = self.extract_featture(crop)
                            names = []
                            distance = []
                            if self.data:
                                for name in self.data.keys():
                                    feature = self.data[name]
                                    cosine = np.dot(emb_vec,feature)/(norm(emb_vec)*norm(feature))
                                    names.append(name)
                                    distance.append(cosine)
                                index_max = np.argmax(distance)
                                id = names[index_max]
                                if distance[index_max] > 0.3:
                                    cv2.putText(frame, id+'{:.2}'.format(distance[index_max]), (x1, y1 - 20),
                                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                                else:
                                    cv2.putText(frame, 'Unknown'.format(distance[index_max]), (x1, y1 - 20),
                                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                            else:
                                cv2.putText(frame, 'Unknown', (x1, y1 - 20),
                                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

                if self.stop_thread:
                    self.stop_thread = False
                    break

                # Emit image data signal
                self.image_data.emit([frame,image_tex])

    def stop(self):
        self.stop_thread = True
        if self.capture is not None:
            self.capture.release()



    def extract_featture(self, crop_face):
        face = Image.fromarray(crop_face)
        TF = base_transform(img_size=112, mode='test')
        face = TF(face)
        hf_face = F.hflip(face).numpy()
        face = face.numpy()
        ft = self.face_recognition_model.inference(face[None])
        hf_ft = self.face_recognition_model.inference(hf_face[None])
        emb_vec = np.concatenate([ft, hf_ft], axis=0)
        return emb_vec



class MyMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        # Load UI file
        ui_file = QFile("form.ui")
        ui_file.open(QFile.ReadOnly)
        uic.loadUi(ui_file, self)
        ui_file.close()

        self.comboBox:QComboBox
        self.msg_name.setText("Field must not be Empty")
        self.msg_name.hide()
        
        self.detection_model = Retinaface_trt('./checkpoint/RetinaFace.trt')
        self.FAS_model = Meta_FAS()

        self.face_recognition_model = Iresnet100('./checkpoint/iresnet1100.trt')
        self.camera_thread = CameraThread(self.detection_model, self.face_recognition_model,self.FAS_model)
        self.is_run_mode = False
        self.mtime = time.ctime(os.path.getmtime('data'))

        # Connect signal/slot for b uttons
        self.comboBox.currentIndexChanged.connect(self.changedMode)
        self.open_webcam_btn.clicked.connect(self.toggle_camera_thread)
        self.data_collect_rbtn.toggled.connect(self.set_image_view)
        self.run_rbtn.toggled.connect(self.set_image_view)
        self.submit_btn.clicked.connect(self.save_face)
        self.update_btn.clicked.connect(self.update_data)

    def changedMode(self):
        self.camera_thread.mode = self.comboBox.currentText()
        # print(self.comboBox.currentText())

    def update_data(self):
        if self.camera_thread is not None:
            self.camera_thread.data = read_data('data')

    def save_face(self):
        if self.camera_thread is not None:
            if self.name_text.text().strip() == "":
                self.msg_name.show()
            else:
                self.camera_thread.save_dir = self.name_text.text()
                self.camera_thread.save_file = True
                self.msg_name.hide()

    def toggle_camera_thread(self):
        if self.camera_thread is not None:
            self.camera_thread.data = read_data('data')
            self.camera_thread.image_data.connect(self.update_image)
            self.camera_thread.mode = self.comboBox.currentText()
            self.camera_thread.start()
            self.open_webcam_btn.setText("Close Webcam")
            
        else:
            self.camera_thread.stop()
            self.camera_thread = None
            self.camera_thread = CameraThread(self.detection_model, self.face_recognition_model,self.FAS_model)
            self.open_webcam_btn.setText("Open Webcam")


    def set_image_view(self):
        if self.data_collect_rbtn.isChecked():
            self.is_run_mode = False
            self.camera_thread.mode = 'collect'
        else:
            self.is_run_mode = True
            self.camera_thread.mode = 'run'
            mtime = time.ctime(os.path.getmtime('data'))
            if self.mtime != mtime: 
               self.camera_thread.data = read_data('data')


    def update_image(self, np_image):
        # Resize and set QImage to QLabel
        img1 = np.copy(np_image[0])
        q_image = self.convert_np_to_qimage(img1)
        img2 = np.copy(np_image[1])
        q_image1 = self.convert_np_to_qimage(img2)
        self.Image1_label.setPixmap(
            QPixmap.fromImage(q_image).scaled(self.Image1_label.width(), self.Image1_label.height(), Qt.KeepAspectRatio))
        self.Image2_label.setPixmap(
            QPixmap.fromImage(q_image1).scaled(self.Image2_label.width(), self.Image2_label.height(), Qt.KeepAspectRatio))


    def convert_np_to_qimage(self, np_image:np.ndarray):
        h, w, ch = np_image.shape
        bytes_per_line = ch * w
        q_image = QImage(np_image.data,w,h,bytes_per_line,QImage.Format_RGB888)
        return q_image


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MyMainWindow()
    window.show()
    sys.exit(app.exec_())