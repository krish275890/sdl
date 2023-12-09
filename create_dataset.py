import os
import pickle
import mediapipe as mp
import cv2
import numpy as np

mp_hands = mp.solutions.hands

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3, max_num_hands=2)  # Set max_num_hands to 2

DATA_DIR = './data'

data = []
labels = []

# Define a fixed length for each hand landmark (assuming there are 21 landmarks per hand)
landmark_length = 21 * 2  # 21 (x, y) coordinates

for dir_ in os.listdir(DATA_DIR):
    dir_path = os.path.join(DATA_DIR, dir_)

    # Check if it's a directory
    if os.path.isdir(dir_path):
        for img_path in os.listdir(dir_path):
            data_aux = []

            x_ = []
            y_ = []

            img = cv2.imread(os.path.join(dir_path, img_path))
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            results = hands.process(img_rgb)
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    for i in range(len(hand_landmarks.landmark)):
                        x = hand_landmarks.landmark[i].x
                        y = hand_landmarks.landmark[i].y

                        x_.append(x)
                        y_.append(y)

                    for i in range(len(hand_landmarks.landmark)):
                        x = hand_landmarks.landmark[i].x
                        y = hand_landmarks.landmark[i].y
                        data_aux.append(x - min(x_))
                        data_aux.append(y - min(y_))

                # Pad or truncate data_aux to ensure a consistent length
                data_aux = data_aux[:landmark_length] + [0] * (landmark_length - len(data_aux))

                data.append(data_aux)
                labels.append(dir_)

# Save the data and labels to a pickle file
with open('data.pickle', 'wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f)