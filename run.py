import cv2
from ultralytics import YOLO
import supervision as sv
import cvzone
import math

cap = cv2.VideoCapture("/dev/video0")  
model = YOLO("best.pt")

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

while True:
    ret, frame = cap.read()

    if not ret:
        print("Error: Can't receive frame (stream end?). Exiting ...")
        break

    detections = model(frame)

    for r in detections:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2-x1, y2-y1
            cvzone.cornerRect(frame, (x1, y1, w, h))

            conf = math.ceil((box.conf[0]*100))/100

            cls = box.cls[0]
            name = cls

            cvzone.putTextRect(frame, f'{name} 'f'{conf}', (max(0,x1), max(35,y1)), scale = 0.5, thickness=1)

    cv2.imshow('Camera Feed', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

