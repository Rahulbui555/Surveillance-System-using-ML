import cv2
import numpy as np

# Load the Haar cascade for face detection
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Create a video capture object to capture video from the default camera
cap = cv2.VideoCapture(0)

# Define the minimum size of the face region for detection
min_face_size = (100, 100)

# Define a list to store the previous positions of the detected face
prev_face_pos = []

# Define a variable to keep track of the number of frames
frame_count = 0

while True:
    # Capture a frame from the video
    ret, frame = cap.read()

    # If there is an error in capturing the video, break from the loop
    if not ret:
        break

    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame using the Haar cascade
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=min_face_size)

    # If a face is detected, extract the face region and calculate the position
    if len(faces) > 0:
        (x, y, w, h) = faces[0]
        face_pos = np.array([x + w/2, y + h/2])

        # Store the current position in the previous positions list
        prev_face_pos.append(face_pos)

        # If the previous positions list is full, remove the oldest position
        if len(prev_face_pos) > 10:
            prev_face_pos.pop(0)

        # Calculate the average displacement of the face over the previous frames
        if len(prev_face_pos) > 1:
            face_disp = np.average(np.diff(prev_face_pos, axis=0), axis=0)
        else:
            face_disp = np.array([0, 0])

        # If the displacement is greater than a threshold, classify the movement direction
        if np.linalg.norm(face_disp) > 20:
            if abs(face_disp[0]) > abs(face_disp[1]):
                if face_disp[0] > 0:
                    direction = 'RIGHT'
                else:
                    direction = 'LEFT'
            else:
                if face_disp[1] > 0:
                    direction = 'DOWN'
                else:
                    direction = 'UP'

            # Draw the movement direction on the frame
            cv2.putText(frame, direction, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

    # Display the color frame
    cv2.imshow('frame', frame)

    # If the 'q' key is pressed, break from the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    cv2.waitKey(50)

# Release the video capture and close all windows
cap.release()
cv2.destroyAllWindows()
