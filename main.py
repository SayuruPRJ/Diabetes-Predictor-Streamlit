import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.impute import KNNImputer
import numpy as np
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from sklearn.utils import class_weight
from keras.regularizers import l2

from keras.models import load_model



from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV


from itertools import combinations
from preprocessing.preprocess import preprocess_for_xgboost
from models.xgbmodel import xgb_model
from preprocessing.preprocess import preprocess_for_logreg
from models.logreg import logreg_model
from preprocessing.preprocess import preprocess_for_NN
from models.neural_network import NN_model


# Read the CSV file
df = pd.read_csv("data\diabetes.csv")


''''
preprocess_for_xgboost(df)


X = df.drop("Outcome", axis=1)
y = df["Outcome"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

scale_pos_weight = y_train.value_counts()[0] / y_train.value_counts()[1]

model = xgb_model(scale_pos_weight)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

'''


'''

df = preprocess_for_logreg(df)


# Separate features and target
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

# Split into train/test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)



best_model = logreg_model()

best_model.fit(X_train,y_train)

'''
df = preprocess_for_NN(df)


# Separate features and target
X = df.drop("Outcome", axis=1)
y = df["Outcome"]


# Split into train/test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

'''
class_weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)

class_weights = dict(enumerate(class_weights))


model = Sequential([
    Dense(16, input_shape=(X_train.shape[1],), activation='relu'),
    Dense(8, activation='relu'),
    Dense(1, activation='sigmoid')  # For binary classification
])

model.compile(
    optimizer=Adam(learning_rate = 0.0005),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=16,
    validation_split=0.1,
    class_weight=class_weights
)
'''


#model = NN_model(X_train, y_train)
model = load_model("DiabetesNN3.keras")

y_pred = model.predict(X_test)

y_pred = (y_pred > 0.5).astype(int)

# Confusion Matrix
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Classification Report (includes precision, recall, f1, accuracy)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))


y_pred_2 = model.predict(X_train)

y_pred_2 = (y_pred_2 > 0.5).astype(int)

# Confusion Matrix
print("Confusion Matrix:")
print(confusion_matrix(y_train, y_pred_2))

# Classification Report (includes precision, recall, f1, accuracy)
print("\nClassification Report:")
print(classification_report(y_train, y_pred_2))

results = X_test.copy()
results["Predicted_Outcome"] = y_pred

