import numpy as np
from sklearn.utils import class_weight
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras import regularizers

def NN_model(X_train,y_train):
        
    class_weights_array = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_train),
        y=y_train
    )

    class_weights = dict(enumerate(class_weights_array))

    # Build the model
    model = Sequential([
        Dense(16, input_shape=(X_train.shape[1],), activation='relu', kernel_regularizer=regularizers.l2(0.001)),
        Dense(8, activation='relu',kernel_regularizer=regularizers.l2(0.001)),
        Dense(1, activation='sigmoid',kernel_regularizer=regularizers.l2(0.001))  # Binary classification
    ])

    # Compile the model
    model.compile(
        optimizer=Adam(learning_rate=0.0005),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    model.fit(
        X_train, y_train,
        epochs=100,
        batch_size=16,
        validation_split=0.1,
        class_weight=class_weights,
        )


    return model