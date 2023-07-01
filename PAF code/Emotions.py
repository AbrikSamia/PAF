import cv2
import numpy as np
from deepface import DeepFace


model = DeepFace.build_model('Emotion')
emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

time_intervals = [(6100, 10316), (10325, 11345), (11370, 21789), (21789, 23625), (23634, 24972), (24972, 25675), (25678, 27151), (27151, 28312), (28312, 36285), (36305, 37511), (37519, 41114), (41114, 42376), (42399, 63124), (63126, 66614), (66644, 93607), (93618, 98356), (98374, 101050), (101050, 105581), (105592, 118038), (118050, 120860), (120867, 144683), (144693, 149493), (149512, 160795), (160795, 161474), (161495, 166455), (166455, 167919), (167951, 194223), (194238, 194643), (194661, 218045), (218060, 220834), (220844, 237371), (237375, 239316), (239335, 249754), (249761, 251214), (251220, 265936), (265948, 267615), (267629, 288448), (288464, 300977), (300999, 303540), (303540, 306344), (306354, 316489), (316516, 338426), (338431, 340647), (340647, 342568), (342609, 349473)]

emotions_vectors = []

cap = cv2.VideoCapture(r'C:\\Users\\HP\\Downloads\\pers2interaction30.mp4')

for interval in time_intervals:
    cap.set(cv2.CAP_PROP_POS_MSEC, interval[0])

    emotions_counter = {label: 0 for label in emotion_labels}
    interval_frames = int((interval[1] - interval[0]) * cap.get(cv2.CAP_PROP_FPS) / 1000)
    current_frame = 0

    while current_frame < interval_frames:
        ret, frame = cap.read()
        if frame is not None:
             resized_frame = cv2.resize(frame, (48, 48), interpolation=cv2.INTER_AREA)
        gray_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)

        img = gray_frame.astype('float32') / 255.0
        img = np.expand_dims(img, axis=-1)
        img = np.expand_dims(img, axis=0)

        preds = model.predict(img)
        emotion_idx = np.argmax(preds)
        emotion = emotion_labels[emotion_idx]

        emotions_counter[emotion] += 1

        current_frame += 1

    dominant_emotions = [label for label in emotion_labels if emotions_counter[label] > 0]
    dominant_emotion_vector = [1 if label in dominant_emotions else 0 for label in emotion_labels]
    print(dominant_emotion_vector)

    emotions_vectors.append(dominant_emotion_vector)

cap.release()

for i, vector in enumerate(emotions_vectors):
    file_path = fr'Data_emotion\30\segment_{i}.npy'
    np.save(file_path, vector)