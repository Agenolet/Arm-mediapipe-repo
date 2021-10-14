# VideoMediapipe01.py
# ESTE SCRIPT EJECUTA UN VIDEO Y MUESTRA LOS LANDMARKS

import cv2
import mediapipe as mp
import tkinter as tk # Útil para obtener las dimensiones del monitor
from time import sleep

#Obtencion de dimensiones del monitor
root = tk.Tk()
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()
print("Monitor actual: " + repr(screen_width) + "x" + repr(screen_height))

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

cap = cv2.VideoCapture ("C:/Visual Code scripts/Arm-mediapipe-repo/Videos/prueba02.mp4")    # ----->>>  SE DEBE EDITAR EN CADA PC

print(cap.get(cv2.CAP_PROP_FRAME_WIDTH))    # Muestro ancho y alto del video a mostrar
print(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# cap.set(3, 640)  #Funciona para camara en vivo o imagenes (creo)     
# cap.set(4, 480)

# print(cap.get(3))
# print(cap.get(4))

if (cap.isOpened() == False):
	print("Error opening the video file")

# Read fps and frame count
else:
	# Get frame rate information. You can replace 5 with CAP_PROP_FPS as well, they are enumerations
	fps = cap.get(5)
	print('Frames per second : ', fps,'FPS')

	# Get frame count. You can replace 7 with CAP_PROP_FRAME_COUNT as well, they are enumerations
	frame_count = cap.get(7)
	print('Frame count : ', frame_count)

with mp_pose.Pose(static_image_mode=False) as pose:
     
     while True:
         ret, frame = cap.read()
         count = 0 # Contador para nombrar diferente a los frames reescalados (resized)

         if ret == False:
             break
        
         # frame = cv2.flip(frame, 1)
         # height, width, layers = frame.shape
         
         # Reescalado de la imagen/imagenes del video
         scale_percent = 70 # percent of original size
         width = int(frame.shape[1] * scale_percent / 100)
         height = int(frame.shape[0] * scale_percent / 100)
         dim = (width, height)

         resized_frame = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA) # resized_frame será el nuevo "frame" que se trabaja

         #  cv2.imwrite("%03d.jpg" % count, resized)
         #  count += 1

         frame_rgb= cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
         results= pose.process (frame_rgb)
        
         if results.pose_landmarks is not None:
                mp_drawing.draw_landmarks(resized_frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS, 
                mp_drawing.DrawingSpec(color=(128,0,250), thickness=2, circle_radius=3), 
                mp_drawing.DrawingSpec(color=(255,255,255), thickness=2))
         #sleep(0.05)
         cv2.imshow("Frame", resized_frame)
         if cv2.waitKey(1) & 0xFF == ord(' '):
            break

cap.release()   #Finaliza la captura
cv2.destroyAllWindows() 
