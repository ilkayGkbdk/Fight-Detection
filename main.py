import os
import numpy as np
import cv2
import tensorflow as tf
from keras._tf_keras.keras.models import load_model, Model
from keras._tf_keras.keras.applications.resnet import ResNet152
from keras._tf_keras.keras.layers import AveragePooling2D, Flatten
import argparse
import time

from keras.src.applications.resnet import preprocess_input

os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model', required=True, help="path to the rnn model")
args = parser.parse_args()

encoder_network = ResNet152(include_top=False, input_shape=(224, 224, 3), weights='imagenet')
for layer in encoder_network.layers:
    layer.trainable = False

out = encoder_network.output
x_model = AveragePooling2D(pool_size=(7, 7), name='avg_pool')(out)
x_model = Flatten()(x_model)

encoder_model = Model(inputs=encoder_network.input, outputs=x_model)
print("Encoder model loaded")

decoder_network = load_model(args.model, compile=False)
print("Decoder network loaded")

x_test_sample = np.random.rand(1, 40, 2048)  # Örnek bir input
output = decoder_network.predict(x_test_sample)
print("Output shape:", output.shape)
print("Output values:", output)


def start(en_model, de_network):
    vs = cv2.VideoCapture(0)
    time.sleep(0.5)
    counter = 0
    x_rnn = np.zeros((40, 2048))
    color = (0, 255, 0)
    classes = ['noFight', 'fight']
    default = 'noFight'
    flag = 1

    while True:
        grabbed, frame = vs.read()
        frame_1 = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_NEAREST)
        f_x = preprocess_input(np.array(frame_1))
        f_x = np.expand_dims(f_x, axis=0)
        feature_map = en_model.predict(f_x)
        x_rnn[counter, :] = np.array(feature_map)
        counter += 1

        if counter % 40 == 0 and flag == 1:
            counter = 0
            flag = 0
            print("prediction time")
            x_test = np.expand_dims(x_rnn, axis=0)
            y_val = de_network.predict(x_test)

            y_val_value = y_val[0][0]
            print(y_val_value)
            if 1e-08 <= y_val_value <= 1e-06:
                class_val = 1  # Fight
            else:
                class_val = 0  # NoFight

            prediction = classes[class_val]
            default = prediction
            if default == 'fight':
                color = (0, 0, 255)
            else:
                color = (0, 255, 0)

        elif counter % 10 == 0 and flag == 0:
            print("prediction time")
            x_test = np.expand_dims(x_rnn, axis=0)
            y_val = de_network.predict(x_test)
            print(y_val)

            y_val_value = y_val[0][0]
            if 1e-08 <= y_val_value <= 1e-06:
                class_val = 1  # Fight
            else:
                class_val = 0  # NoFight

            prediction = classes[class_val]
            default = prediction
            if default == 'fight':
                color = (0, 0, 255)
            else:
                color = (0, 255, 0)

            if counter % 40 == 0:
                counter = 0

        frame = cv2.putText(frame, str(default), (50, 60), cv2.FONT_HERSHEY_COMPLEX, 2, color, 3, cv2.LINE_AA)
        cv2.imshow('Fight Detection Live', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            print("EXITING")
            break

if __name__ == "__main__":
    print("Starting")
    start(encoder_model, decoder_network)
