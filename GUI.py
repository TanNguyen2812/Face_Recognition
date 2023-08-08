import sys
import cv2
import torch
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QRadioButton
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QFile
from PyQt5 import uic
import numpy as np
from MTCNN import MTCNN
import os
from face_regconition_model import iresnet, base_transform
from PIL import Image
import torchvision.transforms.functional as F
import pandas as pd
from read_data import read_data
from numpy.linalg import norm
import time


def crop_face(image, bbox, output_dir, image_name):
    # Get the bounding box coordinates and dimensions
    x1, y1, x2, y2 = bbox.astype(int)
    # Crop the face from the image
    face = image[y1:y2, x1:x2, :]
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # Save the cropped face to file
    filename = os.path.join(output_dir, f"{image_name}.jpg")
    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    cv2.imwrite(filename, face)
    return face


class CameraThread(QThread):
    image_data = pyqtSignal(np.ndarray)

    def __init__(self, face_detection_model, face_recognition_model):
        super().__init__()
        self.capture = None
        self.save_file = False
        self.save_dir = ''
        self.image_name = 1
        self.face_detection_model = face_detection_model
        self.face_recognition_model = face_recognition_model
        self.mode = 'collect'
        self.stop_thread = False
        self.data = None


    def run(self):
        self.capture = cv2.VideoCapture(0)
        while True:
            ret, frame = self.capture.read()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if ret:
                # Run face detection model on frame
                bboxes, _ = self.face_detection_model.detect(frame)
                # Draw bounding boxes on frame
                if bboxes is not None:
                    for bbox in bboxes:
                        print(bboxes)
                        if self.save_file and self.mode == 'collect':
                            path = os.path.join('data', self.save_dir)
                            face = crop_face(frame, bbox, output_dir=path, image_name=str(self.image_name)+'.png')
                            emb_vec = self.extract_featture(face)
                            pd.to_pickle(emb_vec, os.path.join(path, str(self.image_name)+'.pkl'))
                            self.save_file = False
                            self.image_name += 1 # mode
                        x1, y1, x2, y2 = np.clip(bbox.astype(int), a_min=0, a_max=100000)

                        if self.mode == 'run':
                            start = time.time()
                            face = frame[y1:y2, x1:x2]
                            print(face.shape)
                            emb_vec = self.extract_featture(face)
                            names = []
                            distance = []
                            for name in self.data.keys():
                                feature = self.data[name]
                                cosine = np.dot(emb_vec,feature)/(norm(emb_vec)*norm(feature))
                                names.append(name)
                                distance.append(cosine)
                            index_max = np.argmax(distance)
                            id = names[index_max]
                            if distance[index_max] < 0.3:
                                cv2.putText(frame, 'Unknown'.format(distance[index_max]), (x1, y1 - 20),
                                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                            else:
                                cv2.putText(frame, id+'{:.2}'.format(distance[index_max]), (x1, y1 - 20),
                                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                            print('time process', time.time()-start)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

                if self.stop_thread:
                    self.stop_thread = False
                    break

                # Emit image data signal
                self.image_data.emit(frame)

    def stop(self):
        self.stop_thread = True
        if self.capture is not None:
            self.capture.release()



    def extract_featture(self, crop_face):
        face = Image.fromarray(crop_face)
        TF = base_transform(img_size=112, mode='test')
        face = TF(face)
        hf_face = F.hflip(face)
        ft = self.face_recognition_model(face[None].to('cuda'))
        ft = ft[0]
        hf_ft = self.face_recognition_model(hf_face[None].to('cuda'))
        hf_ft = hf_ft[0]
        emb_vec = torch.concat([ft, hf_ft], dim=0)
        emb_vec = emb_vec.detach().cpu().numpy()
        return emb_vec



class MyMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        # Load UI file
        ui_file = QFile("form.ui")
        ui_file.open(QFile.ReadOnly)
        uic.loadUi(ui_file, self)
        ui_file.close()

        self.camera_thread = None
        self.detection_model = MTCNN()
        self.face_recognition_model = iresnet(100)
        self.face_recognition_model.load_state_dict(torch.load('checkpoint/resnet100.pth'))
        self.face_recognition_model.eval()
        self.face_recognition_model.to('cuda')
        self.is_run_mode = False
        self.data = read_data('data')


        # Connect signal/slot for b uttons
        self.open_webcam_btn.clicked.connect(self.toggle_camera_thread)
        self.data_collect_rbtn.toggled.connect(self.set_image_view)
        self.run_rbtn.toggled.connect(self.set_image_view)
        self.submit_btn.clicked.connect(self.save_face)
        self.update_btn.clicked.connect(self.update_data)

    def update_data(self):
        if self.camera_thread is not None:
            self.camera_thread.data = read_data('data')

    def save_face(self):
        if self.camera_thread is not None:
            self.camera_thread.save_dir = self.name_text.text()
            self.camera_thread.save_file = True

    def toggle_camera_thread(self):
        if self.camera_thread is None:
            self.camera_thread = CameraThread(self.detection_model, self.face_recognition_model)
            self.camera_thread.data = read_data('data')
            if self.is_run_mode:
                self.camera_thread.mode = 'run'
            else:
                self.camera_thread.mode = 'collect'
            self.camera_thread.image_data.connect(self.update_image)
            self.camera_thread.start()
            self.open_webcam_btn.setText("Close Webcam")
        else:
            self.camera_thread.stop()
            self.camera_thread = None
            self.open_webcam_btn.setText("Open Webcam")


    def set_image_view(self):
        if self.data_collect_rbtn.isChecked():
            self.is_run_mode = False
            self.image_view = self.Image2_label
        else:
            self.is_run_mode = True
            self.image_view = self.Image1_label


    def update_image(self, np_image):
        # Resize and set QImage to QLabel
        q_image = self.convert_np_to_qimage(np_image)
        self.image_view.setPixmap(
            QPixmap.fromImage(q_image).scaled(self.image_view.width(), self.image_view.height(), Qt.KeepAspectRatio))


    def convert_np_to_qimage(self, np_image):
        h, w, ch = np_image.shape
        bytes_per_line = ch * w
        q_image = QImage(np_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        return q_image


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MyMainWindow()
    window.show()
    sys.exit(app.exec_())