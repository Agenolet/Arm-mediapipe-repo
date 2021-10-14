# Squad_Mediapipe.py
# ESTE SCRIPT DETECTA LA FLEXION DE RODILLA ...

import cv2
import mediapipe as mp
import numpy as np
from math import acos, degrees

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

cap = cv2.VideoCapture ("C:/Visual Code scripts/Arm-mediapipe-repo/Videos/prueba_squad_02.mp4") # ----->>>  SE DEBE EDITAR EN CADA PC

with mp_pose.Pose(static_image_mode=False) as pose:
     
     while True:
         ret, frame = cap.read()
         if ret == False:
             print(ret)
             break
         # frame = cv2.flip(frame, 1)
         # height, width, _ = frame.shape 

         # Reescalado de la imagen/imagenes del video
         scale_percent = 50 # percent of original size
         width = int(frame.shape[1] * scale_percent / 100)
         height = int(frame.shape[0] * scale_percent / 100)
         dim = (width, height)

         resized_frame = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA) # resized_frame será el nuevo "frame" que se trabaja
     
         frame_rgb= cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
         results= pose.process (frame_rgb)
        
         if results.pose_landmarks is not None:
            x1 = int(results.pose_landmarks.landmark[24].x * width)     # Landmark 24 es cadera derecha
            y1 = int(results.pose_landmarks.landmark[24].y * height)
            
            x2 = int(results.pose_landmarks.landmark[26].x * width)     # Landmark 24 es rodilla derecha
            y2 = int(results.pose_landmarks.landmark[26].y * height)

            x3 = int(results.pose_landmarks.landmark[28].x * width)     # Landmark 24 es tobillo derecha
            y3 = int(results.pose_landmarks.landmark[28].y * height)

            # Visualización
            cv2.circle(resized_frame, (x1, y1), 6, (0, 255, 255), 4)
            cv2.circle(resized_frame, (x2, y2), 6, (128, 0, 255), 4)
            cv2.circle(resized_frame, (x3, y3), 6, (255, 191, 0), 4)

         cv2.imshow("Frame", resized_frame)
         if cv2.waitKey(1) & 0xFF == ord(' '):
            break

cap.release()
cv2.destroyAllWindows()