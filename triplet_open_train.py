import json
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import numpy as np
import mediapipe as mp
import cv2
import tensorflow as tf
from tensorflow.keras import layers, models
from keras.models import Model
import random
from sklearn.model_selection import train_test_split

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
    for dir in os.listdir(image_directory):
        new_dir = os.path.join(image_directory, dir)
        for filename in os.listdir(new_dir):
            if not filename.endswith(".pgm"):
                continue

            image_path = os.path.join(new_dir, filename)
            image = read_pgm(image_path)
            images.append(image)
            labels.append(dir)

    return images, labels

def triplet_loss(y_true, y_pred):

    anchor, positive, negative = tf.split(y_pred, 3, axis=1)

    positive_distance = tf.norm(anchor - positive, axis=1)
    negative_distance = tf.norm(anchor - negative, axis=1)

    loss = tf.maximum(positive_distance - negative_distance + 0.2, 0.0)
    return tf.reduce_mean(loss)


def generate_triplets(images, labels):
    triplets = []
    for i in range(len(images)):
        anchor = images[i]
        label = labels[i]
        
        positive = random.choice([img for img, lbl in zip(images, labels) if lbl == label])
        negative = random.choice([img for img, lbl in zip(images, labels) if lbl != label])
        
        triplets.append((anchor, positive, negative))
    return triplets

def preprocess_data(faces):
    new_faces = []
    for face in faces:
        face_resized = cv2.resize(face, (200, 200))
        new_faces.append(face_resized)
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
        layers.Dense(128, activation=None)
    ])
    
    return model


if __name__ == "__main__":
    images, labels = load_data("open_data")
    faces = detect_faces(images)
    faces = preprocess_data(faces)
    triplets = generate_triplets(faces, labels)
    triplets = np.array(triplets)
    print(triplets.shape)

    anchor_images = np.array([triplet[0] for triplet in triplets])
    positive_images = np.array([triplet[1] for triplet in triplets])
    negative_images = np.array([triplet[2] for triplet in triplets])

    anchor_images_train = anchor_images[0:900]
    anchor_images_test = anchor_images[900:]
    positive_images_train = positive_images[0:900]
    positive_images_test = positive_images[900:]
    negative_images_train = negative_images[0:900]
    negative_images_test = negative_images[900:]

    """
    anchor_train, anchor_val, positive_train, positive_val, negative_train, negative_val = train_test_split(
        anchor_images,
        positive_images,
        negative_images,
        test_size=0.2,
        random_state=42
    )

    X_train = triplets
    X_val = triplets[900:]
    y_train = np.zeros(len(triplets))
    y_val = np.zeros(len(triplets[900:]))
    """

    model = build_model()

    anchor_input = layers.Input(shape=(200, 200, 3))
    positive_input = layers.Input(shape=(200, 200, 3))
    negative_input = layers.Input(shape=(200, 200, 3))

    anchor_output = model(anchor_input)
    positive_output = model(positive_input)
    negative_output = model(negative_input)

    triplet_output = layers.concatenate([anchor_output, positive_output, negative_output], axis=1)

    triplet_model = Model(inputs=[anchor_input, positive_input, negative_input], outputs=triplet_output)

    triplet_model.compile(optimizer='adam',
        loss=triplet_loss,
        metrics=['accuracy'])
    
    triplet_model.fit([anchor_images_train, positive_images_train, negative_images_train], np.zeros((len(anchor_images_train), 1)), validation_data=([anchor_images_test, positive_images_test, negative_images_test], np.zeros((len(anchor_images_test), 1))),epochs=5, batch_size=32, shuffle=True)
    #triplet_model.fit([anchor_images, positive_images, negative_images], np.zeros((len(anchor_images), 1)), epochs=5, batch_size=32, shuffle=True)
  
    model.save('open_face_recognition_model.keras')