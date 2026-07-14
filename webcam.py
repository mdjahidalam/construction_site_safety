from ultralytics import YOLO
import cv2

# Load trained model
model = YOLO("models/best.pt")

# Open Laptop Webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Webcam not found!")
    exit()

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # YOLO Prediction
    results = model.predict(frame, conf=0.5)

    # Draw Bounding Boxes
    annotated_frame = results[0].plot()

    # Show Frame
    cv2.imshow("Construction Safety Detection", annotated_frame)

    # Press Q to Exit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()