# Squad_Mediapipe.py
# ESTE SCRIPT DETECTA LA FLEXION DE RODILLA ...

import cv2
import mediapipe as mp
import numpy as np
from math import acos, degrees

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Nombre y ruta de video ORIGINAL
video_path = "C:/Visual Code scripts/Arm-mediapipe-repo/Videos/"    # ----->>>  SE DEBE EDITAR EN CADA PC
video_file_name = "prueba_squad_02"
video_file_extension = ".mp4"
video_file = video_path + video_file_name + video_file_extension
# Carga de archivo
print( "Se analizará: " + video_file + " ...")
cap = cv2.VideoCapture(video_file)

# Datos del video cargado
FPS_original = cap.get(5)  #ej. 24.0 
width_original  = cap.get(3)   # float `width`
height_original = cap.get(4)  # float `height`
resolution_original =  (int(width_original), int(height_original))  #ej. (640, 480)
print("FPS_original:" + str(FPS_original) + " | width_original: " + str(width_original) + " | height_original: " + str(height_original))

# cv2.CAP_PROP_FRAME_WIDTH   # 3
# cv2.CAP_PROP_FRAME_HEIGHT  # 4
# cv2.CAP_PROP_FPS           # 5
# cv2.CAP_PROP_FRAME_COUNT   # 7

# Nombre Y ruta del video generado para guardar como RESULTADO
video_path_result = "C:/Visual Code scripts/Arm-mediapipe-repo/Videos/Videos Resultados/"
video_file_name_result = video_file_name + "_resultado_squad"
video_file_extension_result = video_file_extension
video_file_result = video_path_result + video_file_name_result + video_file_extension_result

# Datos para el video generado para guardar
scale_percent = 50 # Porcentaje de escalado para el video a guardar (será el mismo para el video original a mostrar)
FPS_result = FPS_original   #ej. 24.0
width_result = int(width_original * scale_percent / 100)
height_result = int(height_original * scale_percent / 100)
resolution_result = (width_result, height_result)   #ej. (640, 480)
print("\n-FPS_result: " + str(FPS_result) + "\n-width_result: " + str(width_result) + "\n-height_result: " + str(height_result) + "\n-Resize Escala: " + str(scale_percent) + "\n")

# Creacion de los objetos para el guardado del video prosesado
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
outVideoWriter = cv2.VideoWriter(video_file_result, fourcc, FPS_result, resolution_result) # (name.mp4, fourcc, FPS, resolution)

# Vector de angulos de la rodilla (knee)
V_angles_knee = np.zeros(1)

# Inicio de While True para reproduccion y analisis
with mp_pose.Pose(static_image_mode=False) as pose:
    while True:
        ret, frame = cap.read()
                
        if ret == False:
            break      
        
        # Si la imagen necesita espejarce (flip)
        # frame = cv2.flip(frame, 1)
     
        # Reescalado de la imagen/imagenes del video
        
        # scale_percent = 50  # --> Definido arriba para determinar la escritura del video resultado
        width = int(frame.shape[1] * scale_percent / 100)   # Otra opcion: height, width, layers = frame.shape
        height = int(frame.shape[0] * scale_percent / 100)
        dim = (width, height)

        resized_frame = cv2.resize(frame, dim, interpolation= cv2.INTER_AREA)   # resized_frame será el nuevo "frame" que se trabaja

        frame_rgb = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
        results = pose.process (frame_rgb)

        # Adquiero coordenadas de los marcadores
        if results.pose_landmarks is not None:
            # Landmark 12 es hombro derecha
            x1 = int(results.pose_landmarks.landmark[24].x * width)
            y1 = int(results.pose_landmarks.landmark[24].y * height)

            # Landmark 14 es codo derecha
            x2 = int(results.pose_landmarks.landmark[26].x * width)
            y2 = int(results.pose_landmarks.landmark[26].y * height)

            # Landmark 16 es muñeca derecha
            x3 = int(results.pose_landmarks.landmark[28].x * width)
            y3 = int(results.pose_landmarks.landmark[28].y * height)

            # Calculo de angulo:
            p1 = np.array([x1, y1])
            p2 = np.array([x2, y2])
            p3 = np.array([x3, y3])

            l1 = np.linalg.norm(p2 - p3)
            l2 = np.linalg.norm(p1 - p3)
            l3 = np.linalg.norm(p1 - p2)

            # Calcular el ángulo (teorema del coseno) y lo agrego a V_angles_knee
            angle = degrees(acos((l1**2 + l3**2 - l2**2) / (2 * l1 * l3)))
            V_angles_knee = np.append(V_angles_knee, int(angle))
            
            # Visualización de segmentos de muslo y pierna
            aux_image = np.zeros(resized_frame.shape, np.uint8)

            cv2.line(aux_image, (x1, y1), (x2, y2), (100, 255, 0), 20)
            cv2.line(aux_image, (x2, y2), (x3, y3), (100, 255, 0), 28)
            cv2.line(aux_image, (x1, y1), (x3, y3), (100, 255, 0), 5)
            contours = np.array([[x1, y1], [x2, y2], [x3, y3]])
            cv2.fillPoly(aux_image, pts=[contours], color=(128, 200, 250))

            output = cv2.addWeighted(resized_frame, 1, aux_image, 0.8, 0)   #output es el frame ya prosesado

            cv2.circle(output, (x1, y1), 6, (0, 255, 255), 4)
            cv2.circle(output, (x2, y2), 6, (128, 0, 255), 4)
            cv2.circle(output, (x3, y3), 6, (255, 191, 0), 4)

            cv2.putText(output, str(int(angle)), (x2, y2 - 30), 1, 1.5, (128, 0, 250), 2)   # Agrego el angulo en el video
            cv2.putText(output, "Angulo en grados. Pulse ESPACIO para detener la rep.", (10, height - 10), 4, 0.75, (75, 75, 75), 2) # Agrego info en el video

            # Guardado del video resultante
            outVideoWriter.write(output)
                        
            # Muestro imagenes/video
            cv2.imshow("output", output)    #Muestro video con el triangulo a color

            cv2.imshow("Frame", resized_frame)  #Muestro video original reescalada
            if cv2.waitKey(1) & 0xFF == ord(' '):
                break

# Guardo los angulos medidos
#print("\n ANGULOS: \n")
#print(V_angles_knee)
with open('C:/Visual Code scripts/Arm-mediapipe-repo/Datos/angulos_' + video_file_name + '.txt', 'wb') as f:
    np.savetxt(f, V_angles_knee, delimiter=', ', fmt='%0.1f')

cap.release()
outVideoWriter.release()
cv2.destroyAllWindows()