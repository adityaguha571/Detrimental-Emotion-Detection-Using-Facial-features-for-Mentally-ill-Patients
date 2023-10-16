import cv2
import numpy as np
from keras.models import load_model
import winsound
from twilio.rest import Client
import time

# Load pre-trained model for facial expression recognition
model = load_model("C:\Users\Aditya Guha\OneDrive\Desktop\Detrimental-Emotion-Detection-Using-Facial-features-for-Mentally-ill-Patients-main\Detrimental-Emotion-Detection-Using-Facial-features-for-Mentally-ill-Patients-main\PR691\face_classification-1.0\trained_models\emotion_models\simple_CNN.530-0.65.hdf5")

# Define the set of emotions to be recognized
emotions = ['Angry', 'Sad', 'Neutral', 'Happy', 'Surprise', 'Fear', 'Disgust']

# Initialize the camera or webcam
cap = cv2.VideoCapture(0)

# Twilio credentials and phone numbers
account_sid = 'AC35c7143458fa13b81680a11bc0261c2b'
auth_token = 'f76eac72d4e14e94d0e540152a0cafab'
twilio_phone_number = '+13203355769'
destination_phone_numbers = ['+917044305695', '+918697053380', '+918092861709', '+918942964822', '+917439498488']

# Create a Twilio client
client = Client(account_sid, auth_token)

# Initialize the time of the last alert
last_alert_time = 0

while True:
    # Capture an image of the person's face
    ret, frame = cap.read()

    # Preprocess the image
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (48, 48))  # Update the resizing to (48, 48)
    normalized = resized / 255.0
    reshaped = np.reshape(normalized, (1, 48, 48, 1))  # Update the reshaping to (1, 48, 48, 1)

    # Use the pre-trained model to predict the person's facial expression
    predictions = model.predict(reshaped)[0]
    max_index = np.argmax(predictions)
    if max_index < 0 or max_index >= len(emotions):
        continue  # Skip the current iteration of the loop if max_index is out of range
    emotion = emotions[max_index]

    # Calculate the level of depression, stress, and anxiety based on the person's facial expression
    # You can use your own set of rules or algorithms for this step
    depression = 0
    stress = 0
    anxiety = 0

    if emotion == 'Angry':
        stress += 50
        anxiety += 50
    elif emotion == 'Sad':
        depression += 50
        stress += 40
    elif emotion == 'Happy':
        stress -= 0
        anxiety -= 0
    elif emotion == 'Surprise':
        stress += 30
        anxiety += 20
    elif emotion == 'Fear':
        depression += 30
        stress += 50
        anxiety += 40
    elif emotion == 'Disgust':
        depression += 20
        stress += 30
        anxiety += 30

    # Check if depression, stress, and anxiety levels reach 30
    if depression >= 40 or stress >= 30 or anxiety >= 40:
        current_time = time.time()
        if current_time - last_alert_time >= 30:  # Check if 30 seconds have passed since the last alert
            # Generate an alarm sound
            frequency = 2500  # Set the frequency of the alarm sound
            duration = 2000  # Set the duration of the alarm sound
            winsound.Beep(frequency, duration)

            # Send an alert via SMS to each destination phone number
            '''message = f"Attention: High levels of depression, stress, and anxiety detected!"
            for destination_phone_number in destination_phone_numbers:
                client.messages.create(
                    body=message,
                    from_=twilio_phone_number,
                    to=destination_phone_number
                )

            # Update the time of the last alert
            last_alert_time = current_time'''

    # Display the results
    cv2.putText(frame, f"D: {depression}%, S: {stress}%, A: {anxiety}%", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,
                (255, 255, 255), 2)
    cv2.imshow('Facial Expression Recognition', frame)

    # Exit if the user presses the 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera or webcam and close all windows
cap.release()
cv2.destroyAllWindows()
