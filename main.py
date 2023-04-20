import cv2
from keras.models import load_model
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.vgg16 import preprocess_input
import os


train_data_dir = 'images/training'
validation_data_dir = 'images/Validation'

img_height = 150
img_width = 150

def convert_images_to_grayscale(root_dir):
    """
    Converts all the images in every folder of root_dir to grayscale and replaces the original image.
    """
    for subdir, dirs, files in os.walk(root_dir):
        for file in files:
            file_path = os.path.join(subdir, file)
            if file_path.endswith(".jpg") or file_path.endswith(".png"):
                img = cv2.imread(file_path)
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                cv2.imwrite(file_path, gray)

def testing(train_dir):
    label_ids = {}
    current_id = 0
    for root, dirs, files in os.walk(train_dir):
        for file in files:
            if file.endswith("png") or file.endswith("jpg"):
                label = os.path.basename(root).lower()
                if not label in label_ids.values():
                    label_ids[current_id] = label
                    current_id += 1


def train_facial_recognition_model(train_dir, validation_dir, img_width=150, img_height=150, batch_size=2, epochs=10,num_classes=3):
    # create data generators
    train_datagen = ImageDataGenerator(rescale=1. / 255)
    validation_datagen = ImageDataGenerator(rescale=1. / 255)

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        color_mode='grayscale',
        class_mode='categorical')

    validation_generator = validation_datagen.flow_from_directory(
        validation_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        color_mode='grayscale',
        class_mode='categorical')

    label_ids = {}
    current_id = 0
    for root, dirs, files in os.walk(train_dir):
        for file in files:
            if file.endswith("png") or file.endswith("jpg"):
                label = os.path.basename(root).lower()
                if not label in label_ids.values():
                    label_ids[current_id] = label
                    current_id += 1

    # Create dictionary to store directory names and index values
    # create model architecture
    model = Sequential()

    model.add(Conv2D(32, (3, 3), input_shape=(img_width, img_height, 1), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(256, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())

    model.add(Dense(512, activation='relu'))

    model.add(Dense(num_classes, activation='softmax'))

    # compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'],run_eagerly=True)

    # train model
    model.fit(
        train_generator,
        steps_per_epoch=train_generator.n // batch_size,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=validation_generator.n // batch_size)

    with open("labels.pickle", 'wb') as f:
        pickle.dump(label_ids, f)

    model.save('facial_recognition_model.h5')


#train_facial_recognition_model(train_data_dir,validation_data_dir,num_classes=3,epochs=20)

face_cascade = cv2.CascadeClassifier('casscade/haarcascade_frontalface_default.xml')

def load_facial_recognition_model(model_path):
    """
    Loads the facial recognition model from the given path and returns it.
    """
    model = load_model(model_path)
    return model

model = load_facial_recognition_model("facial_recognition_model.h5")

person_dict = {}
with open("labels.pickle", 'rb') as f:
    person_dict = pickle.load(f)


def recognize_faces_from_camera(model, face_cascade):
    """
    Uses the given facial recognition model to recognize faces from a camera input.

    The function continuously captures frames from the default camera and applies the face detection
    algorithm to each frame. If a face is detected, the face is cropped and resized to match the input
    size of the model. The model is then used to predict the identity of the face, and the predicted
    label is displayed on the frame.

    Press 'q' to exit the program.

    Parameters:
    - model: The trained facial recognition model.
    - face_cascade: The OpenCV cascade classifier for face detection.

    Returns:
    - None
    """
    cap = cv2.VideoCapture(0)
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the grayscale image
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)

        # For each detected face, predict the identity using the loaded model
        for (x, y, w, h) in faces:
            # Crop the face from the frame and resize it to match the input size of the model
            face = cv2.resize(gray[y:y+h, x:x+w], (img_width, img_height))
            face = np.expand_dims(face, axis=2)
            face = np.expand_dims(face, axis=0)

            # Use the model to predict the identity of the face
            probabilities = model.predict(face)[0]
            prediction = np.argmax(probabilities)

            if probabilities[prediction] < 0.95:
                name = "Unknown"
            else:
                name = person_dict[prediction]

            # Draw a rectangle around the detected face and display the predicted label
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, str(name), (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Display the resulting frame
        cv2.imshow('frame', frame)

        # Exit if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

recognize_faces_from_camera(model, face_cascade)

#model = load_facial_recognition_model('facial_recognition_model.h5')

#recognize_faces_from_camera(model, face_cascade)


