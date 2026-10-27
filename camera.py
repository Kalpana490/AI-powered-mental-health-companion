import numpy as np
import cv2
from PIL import Image
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from threading import Thread
import datetime

# Load Haarcascade
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# Disable OpenCL
cv2.ocl.setUseOpenCL(False)

# ---------------------------
# Build Emotion Detection Model
# ---------------------------
emotion_model = Sequential()
emotion_model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48,48,1)))
emotion_model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Dropout(0.25))

emotion_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Dropout(0.25))

emotion_model.add(Flatten())
emotion_model.add(Dense(1024, activation='relu'))
emotion_model.add(Dropout(0.5))
emotion_model.add(Dense(7, activation='softmax'))

emotion_model.load_weights("model.h5")

# Emotion Labels
emotion_dict = {
    0: "Angry",
    1: "Disgusted",
    2: "Fearful",
    3: "Happy",
    4: "Neutral",
    5: "Sad",
    6: "Surprised"
}

# ---------------------------
# Webcam Video Stream Class
# ---------------------------
class WebcamVideoStream:
    def __init__(self, src=0):
        self.stream = cv2.VideoCapture(src, cv2.CAP_DSHOW)
        (self.grabbed, self.frame) = self.stream.read()
        self.stopped = False

    def start(self):
        Thread(target=self.update, args=()).start()
        return self

    def update(self):
        while True:
            if self.stopped:
                return
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
        return self.frame

    def stop(self):
        self.stopped = True


# ---------------------------
# Main VideoCamera Class
# ---------------------------
class VideoCamera(object):

    # Global public variable for Flask API
    last_emotion = "Detecting..."

    def get_frame(self):
        cap = WebcamVideoStream(src=0).start()
        frame = cap.read()

        if frame is None:
            return None, "No Frame"

        frame = cv2.resize(frame, (600, 500))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        detected_emotion = "No Face"

        # Emotion Detection Loop
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (0, 255, 0), 2)

            roi = gray[y:y + h, x:x + w]
            cropped_img = np.expand_dims(
                np.expand_dims(cv2.resize(roi, (48, 48)), -1),
                0
            )

            prediction = emotion_model.predict(cropped_img)
            max_index = int(np.argmax(prediction))

            detected_emotion = emotion_dict[max_index]

            # Put text on frame
            cv2.putText(
                frame,
                detected_emotion,
                (x + 20, y - 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2
            )

        # Update Global Emotion
        VideoCamera.last_emotion = detected_emotion

        # Encode frame as JPEG
        ret, jpeg = cv2.imencode(".jpg", frame)

        return jpeg.tobytes(), detected_emotion
