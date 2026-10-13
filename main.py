import numpy as np
import cv2
import librosa
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
import string
import re
import pickle
from PIL import Image
import io
import tempfile
import os
from ultralytics import YOLO
import google.generativeai as genai
import time
import uuid
import keras
import sys
from flask import Flask, request, jsonify,render_template
from flask_cors import CORS
import serial
import time
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from collections import deque
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button
from scipy.signal import butter, lfilter, lfilter_zi, iirnotch
import eeg_service
from numpy import random
import warnings
warnings.filterwarnings('ignore')
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
dataset = pd.read_csv('dataset.csv')
x_train=dataset.drop(['emotion','timestamp'],axis=1)
y_train=dataset[['emotion']]
le = LabelEncoder()
y_encoded = le.fit_transform(y_train)
X_train, X_test, Y_train, Y_test = train_test_split(x_train,y_encoded, test_size=0.20, random_state=42)
Xtest = np.array(X_test)
Xtest = np.reshape(Xtest, (Xtest.shape[0], Xtest.shape[1], 1))
print(dict(zip(le.classes_, range(len(le.classes_)))))
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Dropout, BatchNormalization
from keras.regularizers import l2
from keras.callbacks import EarlyStopping, LearningRateScheduler
from keras.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
import random
# Define early stopping and learning rate scheduler callbacks
initial_learning_rate = 0.001
def lr_schedule(epoch):
    return initial_learning_rate * (0.1 ** (epoch // 10))

early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
lr_scheduler = LearningRateScheduler(lr_schedule)

# Define the rest of your callbacks
my_callbacks = [
    ModelCheckpoint(filepath='cnnlstm.h5', verbose=1, save_best_only=True, monitor='val_accuracy', mode='max'),
    early_stopping,
    lr_scheduler
]
model = Sequential()

# Add Convolutional and MaxPooling Layers with L2 Regularization
model.add(Conv1D(64, 3, padding="same", input_shape=(Xtest.shape[1], 1), activation='relu', kernel_regularizer=l2(0.01)))
model.add(Conv1D(64, 3, padding="same", activation='relu', kernel_regularizer=l2(0.01)))
model.add(MaxPooling1D(pool_size=2))
model.add(Conv1D(128, 3, padding="same", activation='relu', kernel_regularizer=l2(0.001)))
model.add(Conv1D(128, 3, padding="same", activation='relu', kernel_regularizer=l2(0.001)))
model.add(MaxPooling1D(pool_size=2))

# Add Batch Normalization
model.add(BatchNormalization())

# Add LSTM layer with L2 Regularization
model.add(LSTM(units=100, return_sequences=False, dropout=0.1, kernel_regularizer=l2(0.01)))
model.add(Dropout(0.5))  # Add dropout to reduce overfitting

# Add Dense layer with L2 Regularization
model.add(Dense(units=4, activation='softmax', kernel_regularizer=l2(0.01)))  # 4 classes with softmax activation

# Compile the model with a custom learning rate schedule
opt = Adam(learning_rate=initial_learning_rate)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

model.summary()
model.load_weights('cnnlstm.h5')

# patch keras.src to redirect to keras
sys.modules['keras.src'] = keras
sys.modules['keras.src.legacy'] = keras
app = Flask(__name__)
CORS(app)


from inference_sdk import InferenceHTTPClient

CLIENT = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key="n2sQLiI1teoVwh3jLoY2"
      
)


from ultralytics import YOLO
import cv2
import numpy as np
from scipy.ndimage import zoom
from scipy.signal import welch
# ---------------- CONFIG ----------------
shape_x, shape_y = 48, 48
from tensorflow.keras.models import load_model


emotion_model = load_model("video.h5")
face_detector = YOLO("yolov8n-face.pt")

emotion_labels = ["Angry","Disgust","Fear","Happy","Sad","Surprise","Neutral"]

import time



psychological_map = {
    "angry": "Stress / Irritability",
    "fear": "Anxiety",
    "sad": "Depression",
    "happy": "Normal",
    "neutral": "Normal"
}
face_label = None
face_score = 0.0

eeg_label = None
eeg_score = 0.0

def fuse_emotions(face_label, face_score, eeg_label, eeg_score):
 

    face_weight = 0.8
    eeg_weight = 0.2

    scores = {}
    

    scores[face_label] = face_score * face_weight
    scores[eeg_label] =eeg_score * eeg_weight

    final_emotion = max(scores, key=scores.get)
    final_score = scores[final_emotion]

    return final_emotion, final_score

def detect_disorder(final_emotion):
    return psychological_map.get(final_emotion, "Normal")



def preprocess_face(face):
    face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    face = cv2.resize(face, (shape_x, shape_y))
    face = face.astype("float32") / 255.0
    face = np.reshape(face, (1, 48, 48, 1))
    return face


genai.configure(api_key="AIzaSyCi58V_1rfh5O1b9-NlqOy0UTISTp2eSF4")

generation_config = {
    "temperature": 0.7,
    "top_p": 0.95,
    "top_k": 64,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}

safety_settings = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
]

def band_power(signal, fs, low, high):
    freqs, psd = welch(signal, fs=fs, nperseg=min(len(signal), fs*2))
    idx = (freqs >= low) & (freqs <= high)
    return np.sum(psd[idx])

fs = 256

def featureextraction(eeg):
    delta = band_power(eeg, fs, 0.5, 4)
    theta = band_power(eeg, fs, 4, 8)
    alpha = band_power(eeg, fs, 8, 13)
    beta  = band_power(eeg, fs, 13, 30)
    gamma = band_power(eeg, fs, 30, 45)

   
    frontal_alpha   = alpha * 1.05
    parietal_alpha = alpha * 0.95
    temporal_beta  = beta  * 1.10
    occipital_alpha = alpha * 1.00

  
    total_power = delta + theta + alpha + beta + gamma

    arousal = beta / (alpha + 1e-6)
    valence = np.log(alpha + 1e-6)

    alpha_beta_ratio = alpha / (beta + 1e-6)
    theta_alpha_ratio = theta / (alpha + 1e-6)
    high_freq_ratio = (beta + gamma) / (delta + theta + 1e-6)

    relaxation_index = alpha / (beta + gamma + 1e-6)
    attention_index = beta / (theta + 1e-6)
    meditation_index = (alpha + theta) / (beta + 1e-6)

    
    features = {
        "alpha": alpha,
        "beta": beta,
        "delta": delta,
        "theta": theta,
        "gamma": gamma,

        "frontal_alpha": frontal_alpha,
        "parietal_alpha": parietal_alpha,
        "temporal_beta": temporal_beta,
        "occipital_alpha": occipital_alpha,

        "arousal": arousal,
        "valence": valence,
        "total_power": total_power,

        "alpha_beta_ratio": alpha_beta_ratio,
        "theta_alpha_ratio": theta_alpha_ratio,
        "high_freq_ratio": high_freq_ratio,

        "relaxation_index": relaxation_index,
        "attention_index": attention_index,
        "meditation_index": meditation_index
    }
    return features
feature_order = [
    "alpha", "beta", "delta", "theta", "gamma",
     "arousal", "valence", "total_power",
    "alpha_beta_ratio", "theta_alpha_ratio", "high_freq_ratio",
    "frontal_alpha", "parietal_alpha", "temporal_beta", "occipital_alpha",
    "relaxation_index", "attention_index", "meditation_index"
]
emotion_label_eeg=['angry', 'excited', 'happy', 'sad']
@app.route("/eeg/start", methods=["POST"])
def start_eeg():
    eeg_service.start_eeg()
    return jsonify({"status": "started"})

@app.route("/eeg/stop", methods=["POST"])
def stop_eeg():

    eeg_service.stop_eeg()
    eeg = eeg_service.get_latest_samples(500)
    xtestfeature=featureextraction(eeg)
    X_sample = np.array([xtestfeature[f] for f in feature_order])

    # Single EEG sample
    X_test = np.expand_dims(X_sample, axis=0)

    # Reshape for CNN / CNN-LSTM
    Xtest = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)



    Xtest = np.array(X_test)
    Xtest = np.reshape(Xtest, (Xtest.shape[0], Xtest.shape[1], 1))
    Y_pred = model.predict(Xtest)
    confidence=Y_pred[0][np.argmax(Y_pred)]
    print(confidence)

    Y_pred_classes = np.argmax(Y_pred,axis = 1)
    random.shuffle(emotion_label_eeg) 
    print(emotion_label_eeg[Y_pred_classes[0]])
    print(np.argmax(Y_pred))
    result=emotion_label_eeg[Y_pred_classes[0]]
    print(result)
    eeg_label = result
    eeg_score = confidence
    final_emotion, final_score=fuse_emotions(face_label, face_score, eeg_label, eeg_score)
    disorder = detect_disorder(final_emotion)
    confidence_percent = int(final_score * 100)

    
    print("Final Emotion:", final_emotion)
    print("Confidence:", confidence_percent)
    print("Disorder:", disorder)
    
    return jsonify({
                    'expression': disorder,
                   
                })

   

@app.route("/eeg/data")
def eeg_data():
    return jsonify({
        "samples": eeg_service.get_latest_samples(300)
    })


@app.route('/facial-expression', methods=['POST'])
def facial_expression():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
    file = request.files['image']
    try:
        print(file)
        image_pil = Image.open(file.stream).convert("RGB")

       
        image = np.array(image_pil)

       
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
       
      
        frame=image
        results = face_detector(image, conf=0.5, verbose=False)
        r = results[0]

      
        if r.boxes is None or len(r.boxes) == 0:
            print("No face Detected")
            return jsonify({'error': 'No face/emotion detected'}), 200

       
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                face = frame[y1:y2, x1:x2]

                if face.size == 0:
                    continue

                face_input = preprocess_face(face)
                preds = emotion_model.predict(face_input, verbose=0)
                emotion = emotion_labels[np.argmax(preds)]
                predictedlabel=emotion
                
                print(predictedlabel)
                print(np.argmax(preds))
                print(preds)
                confidence=preds[0][np.argmax(preds)]
                print(confidence)
                face_label = predictedlabel
                face_score = confidence
            
                return jsonify({
                    'expression': predictedlabel,
                   
                })

    except Exception as e:
        print(e)
        return jsonify({'error': str(e)}), 500




chat_history=[]
genai.configure(api_key="AIzaSyDSeRz1RhpFDUkXhzmukTAcq1P3avluMP4")
modelgenai = genai.GenerativeModel(
            model_name="gemini-2.5-flash",
            safety_settings=safety_settings,
            generation_config=generation_config
        )
chat_session = modelgenai.start_chat(history=[])
@app.route("/detect")
def detect():
    return render_template("detect.html")
    
@app.route('/chatbot', methods=['POST'])
def chatbot():
    data = request.json
    print(data)
    if not data or 'message' not in data:
        return jsonify({'error': 'No message provided'}), 400

    user_message = data['message']

    try:
        
        
        chat_history.append({"author": "user", "content": user_message})
        

        response =chat_session.send_message(user_message)
         
        bot_reply = response.text
        chat_history.append({"author": "assistant", "content": bot_reply})
        print(bot_reply)
        

        return jsonify({'reply': bot_reply})
    except Exception as e:
        print(e)
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    app.run(debug=False, host='0.0.0.0', port=5000)