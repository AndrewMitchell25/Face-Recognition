import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import numpy as np
import mediapipe as mp
import cv2
import tensorflow as tf
from tensorflow.keras import layers, models

mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

def preprocess_face(face):
    face_resized = cv2.resize(face, (200, 200))
    return np.array([face_resized])

def video_facial_recognition(model):
    #cap = cv2.VideoCapture(0)
    cap = cv2.VideoCapture('test/test.MP4')

    with mp_face_detection.FaceDetection(min_detection_confidence=0.2) as face_detection:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            rgb_frame = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2RGB)

            results = face_detection.process(rgb_frame)
            if not results.detections:
                continue
            for detection in results.detections:
                b_box = detection.location_data.relative_bounding_box
                ih, iw, _ = rgb_frame.shape
                x, y, w, h = int(b_box.xmin * iw), int(b_box.ymin * ih), int(b_box.width * iw), int(b_box.height * ih)

                if x < 0:
                    w -= x
                    x = 0
                if y < 0:
                    h -= y
                    y = 0
                
                if w == 0 or h == 0:
                    continue

                face = rgb_frame[y:y+h, x:x+w]
                preprocessed_face = preprocess_face(face)

                predictions = model.predict(preprocessed_face, verbose=0)
                predicted_class = np.argmax(predictions, axis=1)
                
                label = 'Andrew' if predicted_class == 0 else 'Unknown'
                color = (0, 255, 0) if predicted_class == 0 else (0, 0, 255)

                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

            cv2.imshow("Face Recognition", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    model = tf.keras.models.load_model('models/face_recognition_model.keras')
    video_facial_recognition(model)