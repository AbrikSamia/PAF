from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
import numpy as np
from sklearn.preprocessing import StandardScaler
import os
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import History
from pympi.Elan import Eaf
def extract_data():
    X=[]
    Y=[]
    for i in [9,10,12,15,18,19,24,26,27,30]:
        file ="C://Users//HP//Documents//Interactions//"+str(i)+"//"+str(i)+".eaf"
        eaf = Eaf(file)
        annots = sorted(eaf.get_annotation_data_for_tier('Trust'))
        repertoire_emotion="C://Users//HP//Documents//Data_emotion//"+str(i)
        repertoire_aus="C://Users//HP//Documents//data_AUs//"+str(i)
        data_e=[]
        data_au=[]


        for num_segment in range(len(os.listdir(repertoire_emotion))):
            filen_name=repertoire_emotion+"//segment_"+str(num_segment)+".npy"
            data_emotion=list(np.load(filen_name))
            data_e.append(data_emotion)
        for num_segment in range(len(os.listdir(repertoire_aus))):
            filen_name=repertoire_aus+"//segment_"+str(num_segment)+".npy"
            data_action=list(np.load(filen_name))
            data_au.append(data_action)

        data_xi=[]
        data_yi=[]

        for k in range(len(data_e)):
            data_xi.append(data_e[k] + data_au[k] )

            tag =annots[k][2]

            if tag=="Neutral":
                y=[0,1,0]
            if tag=="Trusting":
                y=[0,0,1]
            if tag=="Mistrusting":
                y=[1,0,0]

            data_yi.append(y)

            if tag=="Mistrusting":
                data_xi.append(data_e[k]+data_au[k])
                data_yi.append(y)

        X.append(data_xi)
        Y.append(data_yi)

    return (X, Y)

def learn(X, Y):
    for i in range(0, 10):
        X_test = np.array(X[i])
        Y_test = np.array(Y[i])
        X_train = []
        Y_train = []

        for j in range(0, 10):
            if j != i:
                X_train += X[j]
                Y_train += Y[j]

        X_train = np.array(X_train)
        Y_train = np.array(Y_train)

#Modèle RandomForestClassifier
        model_rf = RandomForestClassifier()

# Recherche des meilleurs hyperparamètres
        param_grid = {
        'n_estimators': [100, 200, 300,400,350,150,500],
        'max_depth': [None, 5, 10,15,20,25,30,35,40],
        'min_samples_leaf': [1, 2, 4,8,16,32,64,128,256]
        }
        grid_search = GridSearchCV(model_rf, param_grid, cv=5)
        grid_search.fit(X_train, Y_train)

        best_params = grid_search.best_params_
        best_score = grid_search.best_score_

        model_rf = RandomForestClassifier(**best_params)
        model_rf.fit(X_train, Y_train)

# Prédictions et évaluation la performance
        predictions_rf = model_rf.predict(X_test)
        accuracy_rf = accuracy_score(Y_test, predictions_rf)
        score_rf = model_rf.score(X_test, Y_test)
        print("Random Forest Classifier Score:", score_rf)

# Modèle MLPClassifier
        scaler = StandardScaler()
        X_train_normalized = scaler.fit_transform(X_train)
        X_test_normalized = scaler.transform(X_test)
        model_mlp = MLPClassifier(hidden_layer_sizes=(128, 256), activation='relu', solver='adam', max_iter=100)

# Entraînement du modèle MLPClassifier avec les données normalisées
        model_mlp.fit(X_train, Y_train)

# Prédiction et évaluation du modèle avec les données normalisées
        predictions_mlp = model_mlp.predict(X_test)
        accuracy_mlp = accuracy_score(Y_test, predictions_mlp)
        score_mlp = model_mlp.score(X_test, Y_test)

# Ajout de bruit gaussien aux données d'entraînement
        noise = np.random.normal(0, 0.01 , X_train.shape)
        X_train_noisy = X_train + noise

        print("MLP Classifier Score:", score_mlp)

# Modèle Sequential (Keras)
        model_seq = Sequential()
        history = History()

        model_seq.add(Dense(128, activation='relu', input_shape=(25,)))
        model_seq.add(Dense(256, activation='relu'))
        model_seq.add(Dense(3, activation='softmax'))

        model_seq.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        model_seq.fit(X_train, Y_train, epochs=60, batch_size=32, verbose=False, callbacks=[history])
        loss_seq, accuracy_seq = model_seq.evaluate(X_test, Y_test)
        accuracies_seq = history.history['accuracy']
        losses_seq = history.history['loss']

        plt.plot(range(1, len(accuracies_seq) + 1), accuracies_seq)
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.title('Accuracy vs. Epochs')
        plt.show()

        print("Sequential Model Loss:", loss_seq)
        print("Sequential Model Accuracy:", accuracy_seq)

X, Y = extract_data()
learn(X, Y)