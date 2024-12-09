import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import numpy as np
import mediapipe as mp
import cv2
import tensorflow as tf
from tensorflow.keras import layers, models
import random

mp_face_detection = mp.solutions.face_detection

def read_pgm(pgm_file):
    with open(pgm_file, 'rb') as f:
        header = f.readline().decode().strip()
        if header != 'P5':
            raise ValueError('Invalid PGM file format')

        width = int(f.readline().decode().strip())
        height = int(f.readline().decode().strip())
        max_val = int(f.readline().decode().strip())

        image = np.frombuffer(f.read(), dtype='uint8').reshape((height, width))

    return image

def load_data(image_directory):
    images, labels = [], []
    for filename in os.listdir(image_directory):
        if not filename.endswith(".pgm"):
            continue

        image_path = os.path.join(image_directory, filename)
        image = read_pgm(image_path)
        images.append(image)

        if filename.startswith("Andrew"):
            labels.append(0)
        else:
            labels.append(1)

    return images, labels

def preprocess_data(faces):
    new_faces = []
    for face in faces:
        face_resized = cv2.resize(face, (200, 200))
        face_normalized = face_resized / 255.0
        new_faces.append(face_normalized)
    return new_faces

def detect_faces(images):
    with mp_face_detection.FaceDetection(min_detection_confidence=0.2) as face_detection:
        faces = []
        for image in images:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            results = face_detection.process(rgb_image)
            if results.detections and results.detections[0]:
                detection = results.detections[0]
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
                faces.append(face)

        return faces


def build_model():
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(200, 200, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (5, 5), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(128, activation=None),
        layers.Dense(2, activation='softmax')
    ])
    model.compile(optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'])
    
    return model


if __name__ == "__main__":
    my_images, my_labels = load_data("example_data/Andrew")
    my_faces = detect_faces(my_images)
    my_faces = preprocess_data(my_faces)

    BioID_images, BioID_labels = load_data("example_data/BioID")
    BioID_faces = detect_faces(BioID_images)
    BioID_faces = preprocess_data(BioID_faces)
    random.shuffle(BioID_faces)

    friends_images, friends_labels = load_data("example_data/Friends")
    friends_faces = detect_faces(friends_images)
    friends_faces = preprocess_data(friends_faces)

    X_train = np.array(BioID_faces[0:2] + my_faces[0:2])
    y_train = np.array(BioID_labels[0:2] + my_labels[0:2])
    X_val = np.array(friends_faces + my_faces[2:])
    y_val = np.array(friends_labels + my_labels[2:])

    model = build_model()
    model.fit(X_train, y_train, epochs=10, batch_size=16, validation_data=(X_val, y_val), shuffle=True)
    
    new_model = models.Sequential()
    for layer in model.layers[:-1]:
        new_model.add(layer)

    new_model.save('open_face_recognition_model.keras')
