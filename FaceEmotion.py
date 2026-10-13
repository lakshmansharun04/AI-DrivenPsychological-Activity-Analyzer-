from ultralytics import YOLO
import cv2
import numpy as np
from scipy.ndimage import zoom
from tensorflow.keras.models import load_model

# ---------------- CONFIG ----------------
shape_x, shape_y = 48, 48
emotion_model = load_model("video.h5")
face_detector = YOLO("yolov8n-face.pt")

emotion_labels = ["Angry","Disgust","Fear","Happy","Sad","Surprise","Neutral"]
# ----------------------------------------


def preprocess_face(face):
    face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    face = cv2.resize(face, (shape_x, shape_y))
    face = face.astype("float32") / 255.0
    face = np.reshape(face, (1, 48, 48, 1))
    return face


def show_webcam():
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = face_detector(frame, conf=0.5, verbose=False)
       
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                face = frame[y1:y2, x1:x2]

                if face.size == 0:
                    continue

                face_input = preprocess_face(face)
                preds = emotion_model.predict(face_input, verbose=0)
                emotion = emotion_labels[np.argmax(preds)]

                # Draw bounding box
                cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
                cv2.putText(frame, emotion, (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

                # Display probabilities
                for i, label in enumerate(emotion_labels):
                    cv2.putText(frame,
                        f"{label}: {preds[0][i]:.2f}",
                        (10, 30 + i*20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

        cv2.imshow("YOLO FaceNet Emotion Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    show_webcam()
