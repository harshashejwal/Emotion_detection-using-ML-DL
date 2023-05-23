# -*- cod

#!pip install deepface
from deepface import DeepFace
import cv2
import dlib
import numpy as np
from scipy.spatial import distance as dist

# Define a function to calculate the eye aspect ratio
def eye_aspect_ratio(eye):
    # Compute the Euclidean distances between the two sets of
    # vertical eye landmarks (x, y)-coordinates
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
 
    # Compute the Euclidean distance between the horizontal
    # eye landmark (x, y)-coordinates
    C = dist.euclidean(eye[0], eye[3])
 
    # Compute the eye aspect ratio
    ear = (A + B) / (2.0 * C)
 
    # Return the eye aspect ratio
    return ear

# Initialize the dlib face detector, facial landmark predictor,
# and the threshold for detecting drowsiness
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

#EYE_AR_THRESH = 0.3
EYE_AR_THRESH = 0.2

# Start the video capture
cap = cv2.VideoCapture("emt.mp4")

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
   
size = (frame_width, frame_height)

result = cv2.VideoWriter('filename.avi', 
                         cv2.VideoWriter_fourcc(*'MJPG'),
                         10, size)
                         
while True:
    # Capture a frame from the video feed
    ret, frame = cap.read()
    
    # Convert the frame to grayscale and detect faces
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 0)
    
    # Loop over each face detected
    for rect in rects:
        # Determine the facial landmarks for the face region
        shape = predictor(gray, rect)
        shape = np.array([(shape.part(i).x, shape.part(i).y)
                          for i in range(68)])
        
        # Extract the left and right eye landmarks
        left_eye = shape[36:42]
        right_eye = shape[42:48]
        
        # Calculate the eye aspect ratios for the left and right eyes
        left_ear = eye_aspect_ratio(left_eye)
        right_ear = eye_aspect_ratio(right_eye)
        
        # Calculate the average eye aspect ratio
        ear = (left_ear + right_ear) / 2.0
        
        # Draw the facial landmarks on the frame
        for (x, y) in shape:
            cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)
        
        # Check if the eye aspect ratio is below the threshold
        #if ear < EYE_AR_THRESH:
            # Drowsiness detected, do something
        #    cv2.putText(frame, "Drowsy", (100, 300),
        #                cv2.FONT_HERSHEY_SIMPLEX, 2.7, (0, 0, 255), 10)
        
        emt=DeepFace.analyze(frame,actions=['emotion'],enforce_detection=False)
        #for i in emt:
        print(emt[0]['dominant_emotion'])
        if emt[0]['dominant_emotion'] == 'happy':
            if emt[0]['emotion']['happy'] > 80:
                cv2.putText(frame, 'Happy', (100, 300),cv2.FONT_HERSHEY_SIMPLEX, 2.7, (0, 0, 255), 10)
            elif emt[0]['dominant_emotion'] == 'sad':
                if emt[0]['emotion']['sad'] > 80:
                    cv2.putText(frame, 'Sad', (100, 300),cv2.FONT_HERSHEY_SIMPLEX, 2.7, (0, 0, 255), 10)
            elif emt[0]['dominant_emotion'] == 'surprise':
                if emt[0]['emotion']['surprise'] > 80:
                    cv2.putText(frame, 'Surprise', (100, 300),cv2.FONT_HERSHEY_SIMPLEX, 2.7, (0, 0, 255), 10)
                
        result.write(frame)	

    # Display the frame
    cv2.imshow("Frame", frame)
    
    # Exit the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close the window
cap.release()
result.release()

cv2.destroyAllWindows()

