import time
import pygame
from ultralytics import YOLO

model = YOLO('models/best.pt')

start_time = None
alarm_triggered = False
alarm_duration = 3

pygame.init()  # Tüm Pygame modüllerini başlat
screen = pygame.display.set_mode((640, 480))  # Bir pencere oluştur

results = model.predict(source='videos/IMG_7897.MOV', conf=0.25, stream=True, show=True)

running = True
for result in results:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
            break
    
    if not running:
        break

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

pygame.quit()
