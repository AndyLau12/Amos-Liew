Enginnovate pitching competition 2024  

import cv2
import mediapipe as mp
import math
import time
import serial
from collections import deque

# Function to calculate Eye Aspect Ratio (EAR)
def eye_aspect_ratio(eye_landmarks):
    A = math.dist(eye_landmarks[1], eye_landmarks[5])
    B = math.dist(eye_landmarks[2], eye_landmarks[4])
    C = math.dist(eye_landmarks[0], eye_landmarks[3])
    ear = (A + B) / (2.0 * C)
    return ear

def trecker(average_ear, threshold):
    return threshold > average_ear

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Function to simulate the starting and updating of a stopwatch
def start_stopwatch():
    start_time = time.time()  # Record the start time
    while True:
        elapsed_time = time.time() - start_time
        print(f"\rStopwatch: {elapsed_time:.2f} seconds", end="")
        time.sleep(0.1)  # Update every 0.1 seconds

# Capture video from the webcam
cap = cv2.VideoCapture(0)

# Variables for EAR calculation
reference_ear = None
lowest_ear = None
ear_values = []
lowest_ear_values = []
lowest_ear_start_time = None

# Define start time
start_time = time.time()

# Text settings
font = cv2.FONT_HERSHEY_SIMPLEX
text_color = (0, 255, 0)
text_position = (10, 30)
text_thickness = 1
text_scale = 0.7

# Text settings 2
text_color2 = (0, 0, 255)
text_thickness2 = 2
text_scale2 = 2

# Variables for calculating average EAR
ear_window = deque(maxlen=25)
average_ear = 0.0
ear_update_time = time.time()

#Variables for drowsiness detection
counter=0
alert_triggered = False
drowsiness_detected = False

# Initialize a variable to keep track of when the stopwatch was started
stopwatch_start_time = None

#Define threshold
threshold=0.0 
#Define command 
command=0

# Define the serial port and baud rate
serial_port = 'COM4'  # Change this to match your serial port
baud_rate = 9600

# Initialize the serial connection
ser = serial.Serial(serial_port, baud_rate)
time.sleep(1)  # wait for Arduino to reset

# Main loop to process each frame
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        continue

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(frame_rgb)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            left_eye_indices = [362, 385, 387, 263, 373, 380]
            right_eye_indices = [33, 160, 158, 133, 153, 144]
            left_eye_landmarks = [(int(face_landmarks.landmark[i].x * frame.shape[1]),
                                   int(face_landmarks.landmark[i].y * frame.shape[0])) for i in left_eye_indices]
            right_eye_landmarks = [(int(face_landmarks.landmark[i].x * frame.shape[1]),
                                    int(face_landmarks.landmark[i].y * frame.shape[0])) for i in right_eye_indices]

            left_ear = eye_aspect_ratio(left_eye_landmarks)
            right_ear = eye_aspect_ratio(right_eye_landmarks)
            ear = (left_ear + right_ear) / 2.0

            if time.time() - start_time <= 5:
                ear_values.append(ear)
                cv2.putText(frame, "Obtaining Maximum EAR", (50, 150), font, 1.5, (255, 0, 0), 2, cv2.LINE_AA)

            if reference_ear is not None and lowest_ear_start_time is None:
                lowest_ear_start_time = time.time()

            if lowest_ear_start_time and time.time() - lowest_ear_start_time <= 5:
                lowest_ear_values.append(ear)
                cv2.putText(frame, "Close eyes", (100, 150), font, 1.5, (255, 0, 0), 2, cv2.LINE_AA)

            for landmark in left_eye_landmarks + right_eye_landmarks:
                cv2.circle(frame, landmark, 1, (0, 255, 255), -1)

    if reference_ear is None and time.time() - start_time > 5:
        if ear_values:
            reference_ear = max(ear_values)
            print("Maximum EAR:", reference_ear)

    if lowest_ear_start_time and time.time() - lowest_ear_start_time > 5 and lowest_ear is None:
        if lowest_ear_values:
            lowest_ear = min(lowest_ear_values)
            print("Lowest EAR:", lowest_ear)

    if reference_ear is not None and lowest_ear is not None:
        threshold = lowest_ear + (0.5 * (reference_ear - lowest_ear))
        if ear < threshold:
            text_color = (0, 0, 255)
        else:
            text_color = (0, 255, 0)

    if reference_ear is not None and lowest_ear is not None and time.time() - ear_update_time >=1:
        average_ear = sum(ear_window) / len(ear_window)
        print(f'Average EAR for the last 1 second: {average_ear:.2f}')
        ear_window.clear()
        ear_update_time = time.time()

    if trecker(average_ear, threshold):
        counter += 1

    else:
    # Reset counter and stopwatch if EAR is below threshold
        counter = 0
        stopwatch_start_time = None
    
    # Start the stopwatch if the counter is greater than 1
    if counter >= 1 and stopwatch_start_time is None:
        # Start the stopwatch by recording the current time
        print("Condition met, starting stopwatch...")
        stopwatch_start_time = time.time()
    

    # Calculate and display the stopwatch time if it's running
    if stopwatch_start_time is not None:
        elapsed_time = time.time() - stopwatch_start_time
        cv2.putText(frame, f'Stopwatch: {elapsed_time:.2f}s', (10, 120), font, text_scale, text_color, text_thickness)

        # Check if the stopwatch has reached 0 seconds
        if elapsed_time == 0:
            command=0

    # Check if the stopwatch has reached 5 seconds
        if elapsed_time >= 1 and elapsed_time < 3:
            command=0

     # Check if the stopwatch has reached 5 seconds
        elif elapsed_time >= 3 and elapsed_time < 10:
            command=1
            cv2.putText(frame, 'Driver Alert!', (100, 180), font, text_scale2, text_color2, text_thickness2)
    
    # Check if the stopwatch has reached 10 seconds
        elif elapsed_time >= 10:
            command=2
            cv2.putText(frame, 'Please Wake Up!', (80, 180), font, text_scale2, text_color2, text_thickness2)
        
    else:
    # Reset elapsed time to zero if the stopwatch is not running
        elapsed_time = 0
        command=0

    ear_window.append(ear)

    cv2.putText(frame, f'EAR: {ear:.2f}', text_position, font, text_scale, text_color, text_thickness)
    cv2.putText(frame, f'Avg EAR: {average_ear:.2f}', (10, 60), font, text_scale, text_color, text_thickness)
    cv2.putText(frame, f'Counter: {counter}', (10, 90), font, text_scale, text_color, text_thickness)
    cv2.imshow('MediaPipe FaceMesh - Eyes and Average EAR', frame)

    print(f'command: {command:.2f}')

    #send command to arduino
    ser.write(str(command).encode('utf-8'))


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

if reference_ear is not None and lowest_ear is not None:
    print("Threshold:", threshold)

cap.release()
cv2.destroyAllWindows()



