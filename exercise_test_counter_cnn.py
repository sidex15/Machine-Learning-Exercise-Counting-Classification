import cv2
import mediapipe as mp
import pandas as pd
import numpy as np
import pickle
import time
import tensorflow as tf
from tensorflow.keras.models import load_model
from landmarks import firstRow
from angle_calculator import calcAngle

# Configure GPU usage
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Memory growth needs to be the same across all GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"Found {len(gpus)} GPU(s). GPU acceleration enabled.")
        # Set TensorFlow to only use the first GPU
        if len(gpus) > 1:
            tf.config.set_visible_devices(gpus[0], 'GPU')
    except RuntimeError as e:
        print(f"GPU configuration error: {e}")
else:
    print("No GPU found. Using CPU.")

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

reps_counter=0
reps_duration=0
current_pos=''
prev_pos=''
pTime = time.time()
cTime = 0
count_reset = True

# Change Video path according to the path of the video you want to play
cap = cv2.VideoCapture("./training_videos/Jumping-jack/jumpjack-vid-5.mp4")

# Optional: Set OpenCV to use GPU for video decoding if available
if cv2.cuda.getCudaEnabledDeviceCount() > 0:
    print("OpenCV CUDA backend available")
    # No direct video decoding GPU API in Python OpenCV binding

# Load the trained CNN model using Keras
model = load_model('exercise_cnn_model.h5')

# Load the scaler used during training
with open('cnn_scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Define class names based on the model's training
class_names = ["Unknown", "Situps Down", "Situps UP", "Pushups Down", "Pushups UP", 
               "Planking", "Squat Up", "Squat Down", "Jump Jack Up", "Jump Jack Down"]

# Batch prediction for better GPU utilization
def batch_predict(frames, batch_size=4):
    if len(frames) == 0:
        return []
    
    # Process all frames in batch
    batch_results = model.predict(np.array(frames), batch_size=batch_size, verbose=0)
    return batch_results

# Create a buffer for frame processing
frame_buffer = []
prediction_buffer = []

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("End of video.")
            break
            
        image = cv2.resize(image, (960, 540))
        
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        results = pose.process(image)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        if not results.pose_landmarks:
            cv2.imshow('MediaPipe Pose', image)
            if cv2.waitKey(5) & 0xFF == 27:
                break
            continue
            
        mp_drawing.draw_landmarks(
            image,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(0,215,14), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(255,1,18), thickness=2, circle_radius=1))
        
        try:
            # Extract pose landmarks
            row = np.array([[res.x, res.y] for res in results.pose_landmarks.landmark]).flatten()
            
            # Scale the input data using the same scaler used during training
            X_scaled = scaler.transform([row])
            
            # Reshape for CNN (samples, timesteps, features)
            X_reshaped = X_scaled.reshape(X_scaled.shape[0], 33, 2)
            
            # Add to buffer for batch processing
            frame_buffer.append(X_reshaped[0])
            
            # Process in batches when buffer is full
            if len(frame_buffer) >= 4:  # Adjust batch size based on your GPU memory
                batch_frames = np.array(frame_buffer)
                prediction_buffer = batch_predict(batch_frames)
                frame_buffer = []  # Clear the buffer
            
            # Use the current prediction if available
            if len(prediction_buffer) > 0:
                y_pred_prob = prediction_buffer[0]
                prediction_buffer = prediction_buffer[1:]
            else:
                # Or make a single prediction if buffer is empty
                y_pred_prob = model.predict(X_reshaped, verbose=0)[0]
            
            class_idx = np.argmax(y_pred_prob)
            confidence = y_pred_prob[class_idx]
            
            # Only proceed if confidence is high enough
            if confidence >= 0.95:
                # Get the class name (+1 because class indices in training started from 1)
                class_id = class_idx + 1
                current_pos = class_names[class_id]
                
                # Count reps based on positions
                if class_id == 2:  # Situps Down
                    if count_reset:
                        pTime = time.time()
                        count_reset = False
                    if prev_pos == "Situps UP" and not count_reset:
                        reps_counter += 1
                        cTime = time.time()
                        reps_duration = cTime - pTime
                        count_reset = True
                
                elif class_id == 4:  # Pushups UP
                    if count_reset:
                        pTime = time.time()
                        count_reset = False
                    if prev_pos == "Pushups Down":
                        reps_counter += 1
                        cTime = time.time()
                        reps_duration = cTime - pTime
                        count_reset = True
                
                elif class_id == 6:  # Squat Up
                    if count_reset:
                        pTime = time.time()
                        count_reset = False
                    if prev_pos == "Squat Down":
                        reps_counter += 1
                        cTime = time.time()
                        reps_duration = cTime - pTime
                        count_reset = True
                
                elif class_id == 9:  # Jump Jack Down
                    if count_reset:
                        pTime = time.time()
                        count_reset = False
                    if prev_pos == "Jump Jack Up":
                        reps_counter += 1
                        cTime = time.time()
                        reps_duration = cTime - pTime
                        count_reset = True
            
            # Calculate angle
            a = np.array([results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].x, 
                         results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].y, 
                         results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].z])
            b = np.array([results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP].x, 
                         results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP].y, 
                         results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP].z])
            c = np.array([results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE].x, 
                         results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE].y, 
                         results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE].z])
            
            angle = round(calcAngle(a, b, c), 2)
            angle_text = f"Angle: {angle}"
            reps_duration = round(reps_duration, 2)
            
            # Display information on the frame
            cv2.rectangle(image, (0,0), (250, 40), (245, 117, 16), -1)
            cv2.putText(image, current_pos, (5,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
            cv2.putText(image, f"Reps: {reps_counter}", (10,80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 2, cv2.LINE_AA)
            cv2.putText(image, f"Duration: {reps_duration}", (10,140), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 2, cv2.LINE_AA)
            cv2.putText(image, f"Conf: {confidence:.2f}", (10,170), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1, cv2.LINE_AA)
            cv2.putText(image, angle_text, (10,210), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2, cv2.LINE_AA)
            
            prev_pos = current_pos
            
        except Exception as e:
            print(f"Error: {e}")
            pass
        
        cv2.imshow('MediaPipe Pose', image)
        if cv2.waitKey(5) & 0xFF == 27:
            break

# Process any remaining frames in the buffer
if frame_buffer and len(frame_buffer) > 0:
    batch_predict(np.array(frame_buffer))

cap.release()
cv2.destroyAllWindows()