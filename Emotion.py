import cv2
import numpy as np
import tensorflow as tf
import pandas as pd
from tensorflow.keras.models import load_model

# Load the pre-trained model
model = load_model("C:\Users\Aditya Guha\OneDrive\Desktop\Detrimental-Emotion-Detection-Using-Facial-features-for-Mentally-ill-Patients-main\Detrimental-Emotion-Detection-Using-Facial-features-for-Mentally-ill-Patients-main\PR691\face_classification-1.0\trained_models\emotion_models\simple_CNN.985-0.66.hdf5")
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
emotions = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Neutral', 5: 'Sad', 6: 'Surprise'}

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    # Check if the frame is empty
    if not ret:
        break

    if len(frame.shape) == 3 and frame.shape[2] == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        gray = frame

    # Detect faces in the grayscale frame
    face_cascade = cv2.CascadeClassifier('C:\Users\Aditya Guha\OneDrive\Desktop\Detrimental-Emotion-Detection-Using-Facial-features-for-Mentally-ill-Patients-main\Detrimental-Emotion-Detection-Using-Facial-features-for-Mentally-ill-Patients-main\PR691\haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    # For each face detected, predict the emotion
    for (x, y, w, h) in faces:
        # Extract the face ROI
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        # Preprocess the ROI for the model
        roi_gray = cv2.resize(roi_gray, (48, 48))
        roi_gray = np.expand_dims(roi_gray, axis=-1)
        roi_gray = np.expand_dims(roi_gray, axis=0)
        roi_gray = roi_gray / 255.0

        # Predict the emotion using the model
        preds = model.predict(roi_gray)[0]
        emotion = emotions[np.argmax(preds)]

        # Display the emotion label on the frame
        cv2.putText(frame, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Display the frame
    cv2.imshow('Emotion Recognition', frame)
    cv2.imshow('LIVE', gray)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
cap.release()
cv2.destroyAllWindows() 