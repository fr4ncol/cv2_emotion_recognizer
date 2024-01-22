import cv2
from deepface import DeepFace

cap = cv2.VideoCapture("/dev/video0")  

COLOR = (255,0,255)

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

while True:
    ret, frame = cap.read()

    if not ret:
        print("Error: Can't receive frame (stream end?). Exiting ...")
        break
    try:
        result = DeepFace.analyze(frame, actions=['emotion'])
        print(result)
        for face in result:
            dominant_emotion = face["dominant_emotion"]
            bounding_box = face["region"]
            x1 = bounding_box["x"]
            y1 = bounding_box["y"]
            x2 = bounding_box["x"] + bounding_box["w"] + 10
            y2 = bounding_box["y"] + bounding_box["h"] + 10
            cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR, 1)
            cv2.putText(frame, f"{dominant_emotion}", (x1, y1 - 10), 0, 1, COLOR, 1)
    except:
        print("Face not detected")

    cv2.imshow('Emotion detector', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


