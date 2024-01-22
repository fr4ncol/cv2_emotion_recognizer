import cv2
from deepface import DeepFace

image_path = 'resources/happy_example.jpg'
frame = cv2.imread(image_path)

COLOR = (255, 0, 255)

if frame is None:
    print("Error: Could not open image.")
    exit()

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
except Exception as e:
    print(f"Could not recognize a face.")

cv2.imshow('Emotion detector', frame)
cv2.waitKey(0)
cv2.destroyAllWindows()
