# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 13:22:20 2024

@author: PC
"""

#!/usr/bin/python3
# -*- coding:utf-8 -*-
import dlib
from service.head_pose import HeadPoseEstimator
from service.face_alignment import CoordinateAlignmentModel
from service.face_detector import MxnetDetectionModel
from service.iris_localization import IrisLocalizationModel
import cv2
import numpy as np
from numpy import sin, cos, pi, arctan
from numpy.linalg import norm
import time
from queue import Queue
from threading import Thread
import sys
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tkinter import *
from tkinter.ttk import *
from tkinter.ttk import Progressbar
from tkinter import ttk
from PIL import Image, ImageTk
import tkinter as tk
import sys
win= Tk()
win.configure(background='#d2ffc7')
#win.state('zoomed')
win.attributes("-fullscreen", True)
screen_width = 1920
screen_height = 1080
oran1=screen_width/1280
oran2=screen_height/800
screen_width=screen_width
screen_height=screen_height
global say
listex=[]
listey=[]
say=0
def sayac():
      global say
      say=say+1
      if say==3:
          say=2
          #clear.listex
          #clear.listey
          
          #listex.pop(0)
          #listey.pop(0)

          return say
      return say

import time
# Modeli yükleme
model = load_model('cnn_combined_model_goz5_yeni.h5')
def preprocess_image(eye_region):
    """Göz görüntüsünü CNN modeli için hazırlar"""
    eye_region = cv2.resize(eye_region, (72, 120))  # 30x18 boyutlarına yeniden boyutlandır
    eye_region = eye_region / 255.0  # Normalizasyon
    return eye_region
SIN_LEFT_THETA = 2 * sin(pi / 4)
SIN_UP_THETA = sin(pi / 6)
detector = dlib.get_frontal_face_detector()

def calculate_3d_gaze(frame, poi, scale=256):
    global centers
    starts, ends, pupils, centers = poi

    eye_length = norm(starts - ends, axis=1)
    ic_distance = norm(pupils - centers, axis=1)
    zc_distance = norm(pupils - starts, axis=1)

    s0 = (starts[:, 1] - ends[:, 1]) * pupils[:, 0]
    s1 = (starts[:, 0] - ends[:, 0]) * pupils[:, 1]
    s2 = starts[:, 0] * ends[:, 1]
    s3 = starts[:, 1] * ends[:, 0]

    delta_y = (s0 - s1 + s2 - s3) / eye_length / 2
    delta_x = np.sqrt(abs(ic_distance**2 - delta_y**2))

    delta = np.array((delta_x * SIN_LEFT_THETA,
                      delta_y * SIN_UP_THETA))
    delta /= eye_length
    theta, pha = np.arcsin(delta)

    # print(f"THETA:{180 * theta / pi}, PHA:{180 * pha / pi}")
    # delta[0, abs(theta) < 0.1] = 0
    # delta[1, abs(pha) < 0.03] = 0

    inv_judge = zc_distance**2 - delta_y**2 < eye_length**2 / 4

    delta[0, inv_judge] *= -1
    theta[inv_judge] *= -1
    delta *= scale

    # cv2.circle(frame, tuple(pupil.astype(int)), 2, (0, 255, 255), -1)
    # cv2.circle(frame, tuple(center.astype(int)), 1, (0, 0, 255), -1)

    return theta, pha, delta.T


def draw_sticker(src, offset, pupils, landmarks,
                 blink_thd=0.22,
                 arrow_color=(0, 125, 255), copy=False):
    if copy:
        src = src.copy()

    left_eye_hight = landmarks[33, 1] - landmarks[40, 1]
    left_eye_width = landmarks[39, 0] - landmarks[35, 0]

    right_eye_hight = landmarks[87, 1] - landmarks[94, 1]
    right_eye_width = landmarks[93, 0] - landmarks[89, 0]

    for mark in landmarks.reshape(-1, 2).astype(int):
        cv2.circle(src, tuple(mark), radius=1,
                   color=(0, 0, 255), thickness=-1)

    if left_eye_hight / left_eye_width > blink_thd:
        cv2.arrowedLine(src, tuple(pupils[0].astype(int)),
                        tuple((offset+pupils[0]).astype(int)), arrow_color, 2)

    if right_eye_hight / right_eye_width > blink_thd:
        cv2.arrowedLine(src, tuple(pupils[1].astype(int)),
                        tuple((offset+pupils[1]).astype(int)), arrow_color, 2)

    return src

from datetime import datetime
def main(gpu_ctx=-1):
    global centers
    cap = cv2.VideoCapture(0)

    fd = MxnetDetectionModel("weights/16and32", 0, .6, gpu=gpu_ctx)
    fa = CoordinateAlignmentModel('weights/2d106det', 0, gpu=gpu_ctx)
    gs = IrisLocalizationModel("weights/iris_landmark.tflite")
    hp = HeadPoseEstimator("weights/object_points.npy", cap.get(3), cap.get(4))

    while True:
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame=cv2.resize(frame, (1280, 720))
        gray=cv2.resize(gray, (1280, 720))
        if not ret:
            break

        bboxes = fd.detect(frame)
        faces = detector(frame)
        for face in faces:
            x, y = face.left(), face.top()
            x1, y1 = face.right(), face.bottom()
        for landmarks in fa.get_landmarks(frame, bboxes, calibrate=True):
            # calculate head pose
            _, euler_angle = hp.get_head_pose(landmarks)
            pitch, yaw, roll = euler_angle[:, 0]

            eye_markers = np.take(landmarks, fa.eye_bound, axis=0)
            
            eye_centers = np.average(eye_markers, axis=1)
            # eye_centers = landmarks[[34, 88]]
            
            # eye_lengths = np.linalg.norm(landmarks[[39, 93]] - landmarks[[35, 89]], axis=1)
            eye_lengths = (landmarks[[39, 93]] - landmarks[[35, 89]])[:, 0]

            iris_left = gs.get_mesh(frame, eye_lengths[0], eye_centers[0])
            pupil_left, _ = gs.draw_pupil(iris_left, frame, thickness=1)

            iris_right = gs.get_mesh(frame, eye_lengths[1], eye_centers[1])
            pupil_right, _ = gs.draw_pupil(iris_right, frame, thickness=1)

            pupils = np.array([pupil_left, pupil_right])

            poi = landmarks[[35, 89]], landmarks[[39, 93]], pupils, eye_centers
            theta, pha, delta = calculate_3d_gaze(frame, poi)

            if yaw > 30:
                end_mean = delta[0]
            elif yaw < -30:
                end_mean = delta[1]
            else:
                end_mean = np.average(delta, axis=0)

            if end_mean[0] < 0:
                zeta = arctan(end_mean[1] / end_mean[0]) + pi
            else:
                zeta = arctan(end_mean[1] / (end_mean[0] + 1e-7))

            # print(zeta * 180 / pi)
            # print(zeta)
            if roll < 0:
                roll += 180
            else:
                roll -= 180

            real_angle = zeta + roll * pi / 180
            # real_angle = zeta

            # print("end mean:", end_mean)
            # print(roll, real_angle * 180 / pi)

            R = norm(end_mean)
            offset = R * cos(real_angle), R * sin(real_angle)

            landmarks[[38, 92]] = landmarks[[34, 88]] = eye_centers

            # gs.draw_eye_markers(eye_markers, frame, thickness=1)

            draw_sticker(frame, offset, pupils, landmarks)
            sol_goz_ear=(landmarks[33, 1] - landmarks[40, 1]) / (landmarks[39, 0] - landmarks[35, 0])
            sag_goz_ear=(landmarks[87, 1] - landmarks[94, 1]) / (landmarks[93, 0] - landmarks[89, 0])
            if (sol_goz_ear>0.10 and sag_goz_ear>0.10):
                height_l=int(eye_markers[0][6][1])-int(eye_markers[0][2][1])
                weight_l=int(eye_markers[0][4][0])-int(eye_markers[0][0][0])
                height_r=int(eye_markers[1][6][1])-int(eye_markers[1][2][1])
                weight_r=int(eye_markers[1][4][0])-int(eye_markers[1][0][0])
                
                
                cropped_image10_l = gray[int(eye_markers[0][2][1])-5:int(eye_markers[0][2][1])+height_l+5, int(eye_markers[0][0][0])-5:int(eye_markers[0][0][0])+weight_l+5]
                cropped_image10_r = gray[int(eye_markers[1][2][1])-5:int(eye_markers[1][2][1])+height_r+5, int(eye_markers[1][0][0])-5:int(eye_markers[1][0][0])+weight_r+5]
 
                
                
                left_eye_input = preprocess_image(cropped_image10_l)
                right_eye_input = preprocess_image(cropped_image10_r)
                print(left_eye_input.shape)
                print(right_eye_input.shape)
                X_image_input = np.stack([left_eye_input, right_eye_input], axis=-1).reshape(1, 120, 72, 2)
                X_numeric_input = np.array([[round(pitch, 4), round(yaw, 4), round(roll, 4),
                                             int(centers[0][0]),int(centers[0][1]),
                                             int(centers[1][0]),int(centers[1][1]),
                                             int(pupil_left[0]),int(pupil_left[1]),
                                             int(pupil_right[0]),int(pupil_right[1]),
                                             round(theta[0], 4),round(theta[1], 4),
                                             round(pha[0], 4),round(pha[1], 4),
                                             int(iris_left[0][0]),int(iris_left[1][0]),
                                             int(iris_left[2][0]),int(iris_left[3][0]),
                                             int(iris_left[4][0]),int(iris_left[0][1]),
                                             int(iris_left[1][1]),int(iris_left[2][1]),
                                             int(iris_left[3][1]),int(iris_left[4][1]),
                                             int(iris_right[0][0]),int(iris_right[1][0]),
                                             int(iris_right[2][0]),int(iris_right[3][0]),
                                             int(iris_right[4][0]),int(iris_right[0][1]),
                                             int(iris_right[1][1]),int(iris_right[2][1]),
                                             int(iris_right[3][1]),int(iris_right[4][1]),
                                             int(eye_markers[0][0][0]),int(eye_markers[0][1][0]),
                                             int(eye_markers[0][2][0]),int(eye_markers[0][3][0]),
                                             int(eye_markers[0][4][0]),int(eye_markers[0][5][0]),
                                             int(eye_markers[0][6][0]),int(eye_markers[0][7][0]),
                                             int(eye_markers[0][0][1]),int(eye_markers[0][1][1]),
                                             int(eye_markers[0][2][1]),int(eye_markers[0][3][1]),
                                             int(eye_markers[0][4][1]),int(eye_markers[0][5][1]),
                                             int(eye_markers[0][6][1]),int(eye_markers[0][7][1]),
                                             int(eye_markers[1][0][0]),int(eye_markers[1][1][0]),
                                             int(eye_markers[1][2][0]),int(eye_markers[1][3][0]),
                                             int(eye_markers[1][4][0]),int(eye_markers[1][5][0]),
                                             int(eye_markers[1][6][0]),int(eye_markers[1][7][0]),
                                             int(eye_markers[1][0][1]),int(eye_markers[1][1][1]),
                                             int(eye_markers[1][2][1]),int(eye_markers[1][3][1]),
                                             int(eye_markers[1][4][1]),int(eye_markers[1][5][1]),
                                             int(eye_markers[1][6][1]),int(eye_markers[1][7][1]),
                                             int(x),int(y),int(x1),int(y1),
                                             int(eye_lengths[0]),int(eye_lengths[1])]]).reshape(1, 73)
                
                predictions = model.predict([X_image_input, X_numeric_input])
                x_pred, y_pred = predictions[0]

                # Tahmin edilen değerleri ekrana yazdır
                print(f"Tahmin edilen x: {x_pred}, y: {y_pred}")
                listex.append(x_pred)
                listey.append(y_pred)
                sayac()
                if say==2:
                    xtahmin=sum(listex) / len(listex)
                    ytahmin=sum(listey) / len(listey)
                    
                    if(xtahmin>0 and ytahmin>0 and xtahmin<screen_width and ytahmin<screen_height):                   
                                               
                        print(f"x={xtahmin}, y={ytahmin}")
                        now = datetime.now()
                        print("now =", now)

                        label_ok.place(relx = round(xtahmin/screen_width,4), rely = round(ytahmin/screen_height,4),anchor=CENTER)
                        win.update()  
                        label_ok.pack()    
                    
                               

                        
                        
                    else:
                        print("dışarda")

                    del listex[0:1]
                    del listey[0:1]                 
    main()
   
    


calistir = Button(win,text = "Çalıştır",command = main)

calistir.pack()
#click_btn= PhotoImage(file='goz.png')
#label_ok= Label(image=click_btn)
img = ImageTk.PhotoImage(Image.open('goz.png'))
label_ok = Label(win, image = img,background='#d2ffc7')
#label_ok= Label(win, text= "+", font= ('Helvetica bold', 100),background='red')

label_ok.pack()


win.mainloop()
