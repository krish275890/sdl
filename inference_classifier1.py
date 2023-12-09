import pickle
import cv2
import mediapipe as mp
import numpy as np

model_dict = pickle.load(open('./mlp_model.p', 'rb'))
model = model_dict['model']

cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.3, max_num_hands=2)

labels_dict = {0: 'thanks', 1: 'Yes', 2: 'iloveyou', 3: 'No', 4: 'A', 5: 'Abandon', 6: 'Above', 7: 'Accident', 8: 'Act',
               9: 'Add', 10: 'Address', 11: 'Adult', 12: 'Airplane', 13: 'After', 14: 'Alcohol', 15: 'Alert',
               16: 'Animal', 17: 'Baby', 18: 'Book', 19: 'Calm', 20: 'Cow', 21: 'deer', 22: 'doctor', 23: 'eat',
               24: 'emergency', 25: 'evacuate', 26: 'farm', 27: 'father', 28: 'good', 29: 'hello', 30: 'help',
               31: 'horse', 32: 'may I Use Restroom', 33: 'mine', 34: 'nurse', 35: 'person', 36: 'question',
               37: 'read', 38: 'show', 39: 'small', 40: 'surprise', 41: 'try', 42: 'turtle', 43: 'welcome',
               44: 'what', 45: 'why', 46: 'with', 47: 'write', 48: 'you'}
landmark_length = 21 * 2

predicted_character = ''  # Initialize the variable outside the loop

while True:
    data_aux = []
    x_ = []
    y_ = []

    ret, frame = cap.read()
    H, W, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(frame_rgb)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

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

        data_aux = data_aux[:landmark_length] + [0] * (landmark_length - len(data_aux))

        x1 = int(min(x_) * W) - 10
        y1 = int(min(y_) * H) - 10

        x2 = int(max(x_) * W) - 10
        y2 = int(max(y_) * H) - 10

        prediction = model.predict([np.asarray(data_aux)])
        predicted_character = labels_dict[int(prediction[0])]

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)

    # Display the camera feed
    cv2.imshow('frame', frame)

    # Create a black image as a background for the text box
    text_box = np.zeros((100, W, 3), dtype=np.uint8)

    # Put the predicted character in the text box
    cv2.putText(text_box, predicted_character, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    # Display the text box at the bottom of the screen
    cv2.imshow('Text Box', text_box)

    if cv2.waitKey(1) & 0xFF == 27:  # Press 'Esc' to exit
        break

cap.release()
cv2.destroyAllWindows()
