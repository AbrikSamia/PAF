The project focuses on converting raw data from videos of human-robot interactions into meaningful variables for machine learning models.

## 1. Feature Extraction :
   
The first step of the project is feature extraction, which involves transforming raw data from human-robot interaction videos into significant variables that capture essential and relevant information. In this phase, i worked in a group where each member was responsible for extracting a different type of feature:

Semantics: Extracting meaning from the dialogue or interaction between the human and the robot.

Prosody: Capturing the rhythm, stress, and intonation of speech to detect emotional cues.

Gestures: Identifying and analyzing hand movements and other physical gestures during interactions.

Emotions: Recognizing emotional states expressed by the human participants.

Facial Expressions: Detecting facial expressions and changes in facial features over time.

Distance to the Robot: Measuring the physical distance between the human and the robot during interactions.

My specific role in the project was to extract action units (AUs), which are specific facial muscle movements that form the basis of facial expressions. This process resulted in a feature vector of length 18, capturing the nuances of facial activity during human-robot interaction.

## 2. Data Aggregation :
   
Once features are extracted, the next phase involves data aggregation. This step aggregates and vectorizes the extracted features over segments of the interaction videos. Aggregation serves to condense the extracted data, making it easier to analyze trends and patterns in the interaction.

## 3. Machine Learning Application :
   
In the final stage, the vectorized data is fed into a Random Forest classifier to predict confidence in human-robot interactions. This process involves splitting the dataset into different partitions to train the models on a subset of the data. We tuned hyperparameters using GridSearchCV, experimenting with different values for the number of trees in the forest, maximum depth of the tree, and the minimum number of samples required to be a leaf node. This configuration provided reliable results, and we evaluated model performance by computing the accuracy score on the test set.

![poster PAF png](https://github.com/user-attachments/assets/49b99491-5a5b-4c18-aaab-82cdf56c2d2c)
