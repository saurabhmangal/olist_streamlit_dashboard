import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import streamlit as st
#
# Import Keras modules
#
from keras import models
from keras import layers
from keras.utils import to_categorical
#
# Create the network
#
def neural_net(dataframe):
    network = models.Sequential()
    network.add(layers.Dense(512, activation='relu', input_shape=(4,)))
    network.add(layers.Dense(3, activation='softmax'))
    #
    # Compile the network
    #
    network.compile(optimizer='rmsprop',
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])
    #
    # Load the iris dataset
    #
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target
    st.write(X)
    st.write(y)
    #
    # Create training and test split
    #
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
    #
    # Create categorical labels
    #
    train_labels = to_categorical(y_train)
    test_labels = to_categorical(y_test)
    #
    # Fit the neural network
    #
    network.fit(X_train, train_labels, epochs=20, batch_size=40)

    #
    # Get the accuracy of test data set
    #
    test_loss, test_acc = network.evaluate(X_test, test_labels)
    #
    # Print the test accuracy
    #
    st.write('Test Accuracy: '+str(test_acc)+ '\nTest Loss: '+str(test_loss))