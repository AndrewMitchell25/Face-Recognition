import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import numpy as np
import mediapipe as mp
import cv2
import tensorflow as tf
from numpy.linalg import norm

mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

def preprocess_face(face):
    face_resized = cv2.resize(face, (200, 200))
    face_normalized = face_resized / 255.0

    return np.array([face_normalized])

def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (norm(vec1) * norm(vec2))

def verify_face(embedding, known_embeddings, threshold=.7):
    for name, known_embedding in known_embeddings.items():
        similarity = cosine_similarity(embedding.tolist()[0], known_embedding[0])
        #print(similarity)
        if similarity > threshold:
            #print(name)
            return name
    return False

def video_facial_recognition(model, known_embeddings, video_path, test_percentage):
    #cap = cv2.VideoCapture(0)
    cap = cv2.VideoCapture(video_path)

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

                embedding = model.predict(preprocessed_face, verbose=0)
                
                name = verify_face(embedding, known_embeddings)

                if name:
                    label = name
                    color = (0, 255, 0)
                else:
                    label = 'Unknown'
                    color = (0, 0, 255)

                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

            cv2.imshow("Face Recognition", frame)

            if test_percentage:
                key = cv2.waitKey(0) & 0xFF
                # Correctly identify known face and unknown face
                if key == ord('1'):
                    test_percentage['true-positive'] += 1
                    test_percentage['true-negative'] += 1
                # Correctly identify known face but fail to identify unknown face
                if key == ord('2'):
                    test_percentage['true-positive'] += 1
                    test_percentage['false-positive'] += 1
                # Fail to identify known face and correctly identify unknown face
                if key == ord('3'):
                    test_percentage['false-negative'] += 1
                    test_percentage['true-negative'] += 1
                # Fail to identify known face and unknown face
                if key == ord('4'):
                    test_percentage['false-negative'] += 1
                    test_percentage['false-positive'] += 1
                else:
                    continue
            else:
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break


        cap.release()
        cv2.destroyAllWindows()

def generate_known_embeddings(model, image_dir):
    known_embeddings = {}
    with mp_face_detection.FaceDetection(min_detection_confidence=0.2) as face_detection:
        for filename in os.listdir(image_dir):
            image = cv2.imread(os.path.join(image_dir, filename))

            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            rgb_image = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2RGB)

            results = face_detection.process(rgb_image)

            for detection in results.detections:
                b_box = detection.location_data.relative_bounding_box
                ih, iw, _ = rgb_image.shape
                x, y, w, h = int(b_box.xmin * iw), int(b_box.ymin * ih), int(b_box.width * iw), int(b_box.height * ih)

                if x < 0:
                    w -= x
                    x = 0
                if y < 0:
                    h -= y
                    y = 0
                
                if w == 0 or h == 0:
                    continue

                face = rgb_image[y:y+h, x:x+w]
                preprocessed_face = preprocess_face(face)

                embedding = model.predict(preprocessed_face, verbose=0)
                #print(embedding, filename)
                known_embeddings[filename.split(".")[0]] = embedding

    return known_embeddings

if __name__ == "__main__":
    model = tf.keras.models.load_model('models/open_face_recognition_model.keras')
    known_embeddings = generate_known_embeddings(model, 'Test_Known_Faces')
    #test_percentage = {'true-positive': 0, 'false-positive': 0, 'true-negative': 0, 'false-negative': 0}
    test_percentage = None 
    for video in os.listdir("test"):
        if video.endswith("MP4"):
            video_facial_recognition(model, known_embeddings, f'test/{video}', test_percentage)
    if test_percentage:
        print(test_percentage)