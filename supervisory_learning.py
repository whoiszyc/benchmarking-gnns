import keras
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from data_test_case import case33_tieline, case33_tieline_DG
from NN_model import DNN_TieLine, DNN_VarCon


# ======================= reshape the data =====================
dt = pd.read_csv("trajectory_BC__2021_01_04_13_59.csv", converters={'state': eval, 'action': eval})

n_sample = dt.shape[0]
x = np.reshape(dt['state'].iloc[0], (1, 37))
y = np.reshape(dt['action'].iloc[0], (1, 6))
for i in range(1, n_sample):
    print(i)
    x = np.append(x, np.reshape(dt['state'].iloc[i], (1, 37)), axis=0)
    y = np.append(y, np.reshape(dt['action'].iloc[i], (1, 6)), axis=0)
x = pd.DataFrame(x)
y = pd.DataFrame(y)

x.to_csv('trajectory_BC__2021_01_04_13_59_X.csv', index=False)
y.to_csv('trajectory_BC__2021_01_04_13_59_Y.csv', index=False)

# # ============ using supervised learning for neural network tuning==============
train_split = 0.6
X = pd.read_csv("trajectory_BC__2021_01_04_13_59_X.csv")
Y = pd.read_csv("trajectory_BC__2021_01_04_13_59_Y.csv")


n_sample = X.shape[0]

X = X.to_numpy()
Y = Y.to_numpy()

indecs = np.random.permutation(len(X))
X = X[indecs]
Y = Y[indecs]

idx_action_0 = np.where(Y.argmax(axis=1) == 0)[0]
X0 = X[idx_action_0[:int(n_sample * 0.125)]]
Y0 = Y[idx_action_0[:int(n_sample * 0.125)]]

idx_action_o = np.where(Y.argmax(axis=1) != 0)[0]
Xo = X[idx_action_o]
Yo = Y[idx_action_o]

X = np.concatenate((X0, Xo), axis=0)
Y = np.concatenate((Y0, Yo), axis=0)

n_sample_1 = X.shape[0]

x_train = X[:int(n_sample_1 * train_split)]
y_train = Y[:int(n_sample_1 * train_split)]
print("Sample for training: {}".format(x_train.shape[0]))
# testing data from 80% to 100%
x_test = X[int(n_sample_1 * train_split):]
y_test = Y[int(n_sample_1 * train_split):]
print("Sample for testing: {}".format(x_test.shape[0]))


# # ------------training verification for continuous actions: var dispatch -------------
# model = Sequential()
# model.add(Dense(64, input_shape=(33,), activation="tanh"))  # sigmoid
# model.add(Dense(128, activation="tanh"))
# model.add(Dense(128, activation="tanh"))
# # model.add(Dense(128, activation="tanh"))
# # model.add(Dense(128, activation="tanh"))
# model.add(Dense(6, activation="tanh"))
# model.compile(loss="mean_squared_error", optimizer=Adam(lr=0.001), metrics=['mean_squared_error'])  # categorical_crossentropy，  kullback_leibler_divergence
# model.fit(x_train, y_train, batch_size=32, epochs=300, verbose=2)
# y_hat = model.predict(x_test, batch_size=32)
# for i in range(6):
#     plt.rcParams.update({'font.family': 'Arial'})
#     plt.figure(figsize=(9, 7))
#     plt.plot(y_hat[:,i])
#     plt.plot(y_test[:,i],'-.')
#     plt.title("Var {}".format(i))

# # ------------training verification for discrete actions: tie-line actions -------------
model = Sequential()
model.add(Dense(64, input_shape=(X.shape[1],), activation="relu"))  # sigmoid
model.add(Dense(128, activation="relu"))
model.add(Dense(64, activation="relu"))
model.add(Dense(6, activation="softmax"))
model.compile(loss="categorical_crossentropy", optimizer=Adam(lr=0.001), metrics=['categorical_accuracy'])  # categorical_crossentropy，  kullback_leibler_divergence
model.fit(x_train, y_train, batch_size=32, epochs=100, verbose=2)

y_hat = model.predict(x_test, batch_size=32)
action_hat = y_hat.argmax(axis=1)
action_real = y_test.argmax(axis=1)
accuracy = action_hat == action_real
accuracy = accuracy.sum()/len(y_test)
print("Predicted action accuracy {}".format(accuracy))
