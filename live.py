import time
import pygame
from ultralytics import YOLO

model = YOLO('models/best_1.pt')

start_time = None
alarm_triggered = False
alarm_duration = 3 

pygame.mixer.init()

results = model.predict(source=0, conf=0.4, stream=True, show=True)

for result in results:
    boxes = result.boxes
    powderless_detected = False
    powdered_detected = False
    
    for box in boxes:
        class_name = model.names[int(box.cls)]
        if class_name in ['powderless', 'powderlesss']:
            powderless_detected = True
        elif class_name in ['powdered', 'powdereds']:
            powdered_detected = True
    
    if powderless_detected and not alarm_triggered:
        if start_time is None:
            start_time = time.time()
        elif time.time() - start_time >= alarm_duration:
            print("Alarm tetiklendi, ses çalınıyor...")
            pygame.mixer.music.load('alarm.mp3') 
            pygame.mixer.music.play(loops=-1) 
            alarm_triggered = True
    
    if powdered_detected and alarm_triggered:
        pygame.mixer.music.stop()  
        print("Alarm durduruldu, powdered tespit edildi.")
        alarm_triggered = False 

    if not powderless_detected and not alarm_triggered:
        start_time = None  

pygame.mixer.quit()
