import os
import numpy as np
import tensorflow as tf
from keras._tf_keras.keras.layers import Dense,BatchNormalization,LSTM,Dropout,Input
from keras._tf_keras.keras.models import Model
from keras._tf_keras.keras.callbacks import ReduceLROnPlateau
from keras._tf_keras.keras.optimizers import Adam
from keras import backend
from keras._tf_keras.keras import regularizers
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-d", ""'--featuresPath', required = True, help = "path to the features folder")
parser.add_argument("-s", ""'--savingPath', required = True, help = "path to the saving folder for model")

args = parser.parse_args()
data_x = os.path.join(args.featuresPath, "data_x_ext.npy")
data_y = os.path.join(args.featuresPath, "data_y.npy")

x_train = np.load(data_x)
y_train = np.load(data_y)

# transpose
x_r = []
for i in range(x_train.shape[0]):
    x_temp = x_train[i, :, :]
    x_t = x_temp.T
    x_r.append(x_t)

x_train = np.array(x_r)

backend.clear_session()

def rnn_model():
    x_input = Input(shape=(40, 2048))
    x = LSTM(units=1024, return_sequences=True, dropout=0.4)(x_input)
    x = LSTM(units=512, return_sequences=False, dropout=0.3)(x)
    x = Dense(units=256, activation='relu', kernel_regularizer=regularizers.l2())(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    x = Dense(units=256, activation='relu', kernel_regularizer=regularizers.l2())(x)
    x = BatchNormalization()(x)
    x = Dense(units=1, activation='sigmoid')(x)

    adam = Adam(learning_rate=0.005, decay=1e-6)

    model = Model(inputs=x_input, outputs=x)
    model.summary()
    model.compile(
        optimizer=adam,
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return model

model = rnn_model()

cb = [ReduceLROnPlateau(patience=5, verbose=1),]
output = model.fit(
    x_train,
    y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.2,
    verbose=1,
    callbacks=cb,
    shuffle=True,
)

model.save(os.path.join(args.savingPath, "fightD_model.h5"))
model.evaluate(x_train, y_train)

acc_value = output.history['accuracy']
val_acc_value = output.history['val_accuracy']
loss_value = output.history['loss']
val_loss_value = output.history['val_loss']

epochs = range(1, len(acc_value) + 1)

plt.plot(epochs, acc_value, 'b', label='Training accuracy')
plt.plot(epochs, val_acc_value, 'r', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend(loc='upper right')

plt.figure()

plt.plot(epochs, loss_value, 'b', label='Training loss')
plt.plot(epochs, val_loss_value, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.legend(loc='upper right')

plt.show()
