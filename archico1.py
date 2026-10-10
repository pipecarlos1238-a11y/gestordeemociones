#!/usr/bin/env python3
import cv2
import numpy as np
import os
import json
import math
from datetime import datetime

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import Image, ImageFormat

base_options = python.BaseOptions(model_asset_path='face_landmarker.task')
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    num_faces=1,
    min_face_detection_confidence=0.5,
    min_face_presence_confidence=0.5,
    min_tracking_confidence=0.5,
    running_mode=vision.RunningMode.VIDEO
)
detector = vision.FaceLandmarker.create_from_options(options)

# Índices de MediaPipe para la boca
LABIO_EXTERNO = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37]
LABIO_INTERNO = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415, 310, 311, 312, 13, 82, 81, 80, 191]

def calcular_distancia(p1, p2):
    return math.hypot(p2[0] - p1[0], p2[1] - p1[1])

def calcular_features_avanzadas(face_landmarks, ancho_img, alto_img):
    if not face_landmarks: return None
    
    def get_p(idx):
        return (int(face_landmarks[idx].x * ancho_img), int(face_landmarks[idx].y * alto_img))
    
    try:
    
        ext_izq = get_p(61)
        ext_der = get_p(291)
        ext_sup = get_p(0)
        ext_inf = get_p(17)
        
    
        int_izq = get_p(78)
        int_der = get_p(308)
        int_sup = get_p(13)
        int_inf = get_p(14)
        
        # Ojos 
        ojo_izq_ext = get_p(33)
        ojo_der_ext = get_p(263)
        distancia_ojos = calcular_distancia(ojo_izq_ext, ojo_der_ext)
        
        if distancia_ojos == 0: return None

        # Ancho de la boca normalizado
        ancho_boca = calcular_distancia(ext_izq, ext_der)
        ancho_norm = ancho_boca / distancia_ojos
        
        # (Apertura externa)
        alto_ext = calcular_distancia(ext_sup, ext_inf)
        mar_externo = alto_ext / ancho_boca if ancho_boca > 0 else 0
        
        #(Apertura interna - detecta si hay espacio entre los dientes)
        ancho_int = calcular_distancia(int_izq, int_der)
        alto_int = calcular_distancia(int_sup, int_inf)
        mar_interno = alto_int / ancho_int if ancho_int > 0 else 0
        
        # Curvatura de la sonrisa 
        centro_y = (ext_sup[1] + ext_inf[1]) / 2
        comisuras_y = (ext_izq[1] + ext_der[1]) / 2
        curva_sonrisa = (centro_y - comisuras_y) / distancia_ojos
        
        # Tensión Labial 
        grosor_sup = calcular_distancia(ext_sup, int_sup)
        grosor_inf = calcular_distancia(int_inf, ext_inf)
        tension_labial = (grosor_sup + grosor_inf) / ancho_boca if ancho_boca > 0 else 0

        features = [
            ancho_norm,     
            mar_externo,    
            mar_interno,    
            curva_sonrisa,  
            tension_labial  
        ]
        
        return features
        
    except Exception as e:
        return None

def capturar_datos_entrenamiento():
    camara = cv2.VideoCapture(0)
    camara.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    camara.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    datos = {0: [], 1: [], 2: [], 3: [], 4: []}
    nombres = {0: "Neutro", 1: "Felicidad", 2: "Tristeza", 3: "Enojo", 4: "Sorpresa"}
    frame_idx = 0
    
    print("\n" + "="*50 + "\nCAPTURADOR DE EMOCIONES AVANZADO\n" + "="*50)
    print("Presiona los números (0-4) para capturar muestras.")
    print("Presiona 'q' para guardar y salir.\n")
    
    while True:
        ret, frame = camara.read()
        if not ret: break
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        alto, ancho = frame_rgb.shape[:2]
        
        mp_image = Image(image_format=ImageFormat.SRGB, data=frame_rgb)
        results = detector.detect_for_video(mp_image, frame_idx)
        frame_idx += 1
        
        features = None
        
        if results.face_landmarks:
            face_landmarks = results.face_landmarks[0]
            features = calcular_features_avanzadas(face_landmarks, ancho, alto)
            
            # DIBUJAR LOS LABIOS
            def obtener_puntos(indices):
                return np.array([[(int(face_landmarks[i].x * ancho), int(face_landmarks[i].y * alto))] for i in indices], dtype=np.int32)
            
            puntos_ext = obtener_puntos(LABIO_EXTERNO)
            puntos_int = obtener_puntos(LABIO_INTERNO)
            
            cv2.polylines(frame, [puntos_ext], True, (0, 255, 0), 2)
            cv2.polylines(frame, [puntos_int], True, (255, 255, 0), 1)
            
            # Mostrar métricas en vivo
            if features:
                cv2.putText(frame, f"Curva: {features[3]:.3f} | Tension: {features[4]:.3f}", 
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                cv2.putText(frame, f"Ap. Int: {features[2]:.3f} | Ap. Ext: {features[1]:.3f}", 
                            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        
        overlay = frame.copy()
        cv2.rectangle(overlay, (5, 80), (200, 220), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        
        y_text = 105
        for key, name in nombres.items():
            cv2.putText(frame, f"[{key}] {name}: {len(datos[key])}", (15, y_text),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
            y_text += 25
            
        cv2.imshow('Deteccion de Boca Avanzada', frame)
        
        tecla = cv2.waitKey(1) & 0xFF
        if tecla == ord('q'): break
        elif ord('0') <= tecla <= ord('4') and features:
            emocion_idx = int(chr(tecla))
            datos[emocion_idx].append(features)
            print(f"[{nombres[emocion_idx]}] Muestra #{len(datos[emocion_idx])} capturada")

    camara.release()
    cv2.destroyAllWindows()
    
    # GUARDAR DATOS
    carpeta_datos = "datos_emociones"
    os.makedirs(carpeta_datos, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archivo_csv = f"{carpeta_datos}/dataset_avanzado_{timestamp}.csv"
    
    total = sum(len(d) for d in datos.values())
    if total > 0:
        with open(archivo_csv, 'w') as f:
            f.write("ancho_norm,mar_externo,mar_interno,curva_sonrisa,tension_labial,label\n")
            for label, d_list in datos.items():
                for feat in d_list:
                    f.write(f"{feat[0]:.6f},{feat[1]:.6f},{feat[2]:.6f},{feat[3]:.6f},{feat[4]:.6f},{label}\n")
        
        metadata = {
            "timestamp": timestamp, "total": total,
            "features": ["ancho_norm", "mar_externo", "mar_interno", "curva_sonrisa", "tension_labial"],
            "labels": nombres
        }
        with open(f"{carpeta_datos}/metadata_{timestamp}.json", 'w') as f:
            json.dump(metadata, f, indent=2)
            
        print(f"\n¡Listo! {total} muestras guardadas en la carpeta 'datos_emociones'.")
    else:
        print("\nNo se guardaron datos porque no se capturó ninguna muestra.")

if __name__ == "__main__":
    capturar_datos_entrenamiento()  