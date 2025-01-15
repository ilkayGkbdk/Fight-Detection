import os
import numpy as np
import tensorflow as tf
from keras._tf_keras.keras.layers import AveragePooling2D, Flatten
from keras._tf_keras.keras.models import Model
from PIL import Image
import tqdm
from keras.src.applications.resnet import ResNet152

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

class DataHandler:
    def __init__(self, data_path, classes, max_frames, img_shape, channels, saving_dir):
        self.data_path = data_path
        self.classes = classes
        self.seq_length = max_frames
        self.width = img_shape[0]
        self.height = img_shape[1]
        self.channels = channels
        self.saving_dir = saving_dir

        self.base_model = ResNet152(include_top=False, input_shape=(224, 224, 3), weights='imagenet')
        for layer in self.base_model.layers:
            layer.trainable = False

        self.output = self.base_model.output
        self.x_model = AveragePooling2D(pool_size = (7, 7), name = 'avg_pool')(self.output)
        self.x_model = Flatten()(self.x_model)

        self.model = Model(inputs = self.base_model.input, outputs = self.x_model)
        print(self.model.summary())

    def get_frame_sequence(self, path):
        flag = 1
        total_frames = os.listdir(path)
        arr = np.zeros((224,224,3,40))
        if len(total_frames) >= 160:
            counter = 0
            for i in range(1,160,4):
                x = Image.open(os.path.join(path,str(i) + '.jpg')).resize((224,224))
                x = np.array(x)
                arr[:,:,:,counter] = x
                counter+=1
                if counter >= self.seq_length:
                    break

        elif (len(total_frames) >= 120) and (len(total_frames) < 160):
            counter = 0
            for i in range(1,120,3):
                x = Image.open(os.path.join(path,str(i) + '.jpg')).resize((224,224))
                x = np.array(x)
                arr[:,:,:,counter] = x
                counter+=1
                if counter >= self.seq_length:
                    break

        elif (len(total_frames) >= 99) and (len(total_frames) < 120):
            counter = 0
            for i in range(0,40,2):
                x = Image.open(os.path.join(path,str(i) + '.jpg')).resize((224,224))
                x = np.array(x)
                arr[:,:,:,counter] = x
                counter+=1
            for i in range(41,99,3):
                x = Image.open(os.path.join(path,str(i) + '.jpg')).resize((224,224))
                x = np.array(x)
                arr[:,:,:,counter] = x
                counter += 1
                if counter >= self.seq_length:
                    break

        elif (len(total_frames) >= 80) and (len(total_frames) < 98):
            counter = 0
            for i in range(0,80,2):
                x = Image.open(os.path.join(path,str(i) + '.jpg')).resize((224,224))
                x = np.array(x)
                arr[:,:,:,counter] = x
                counter += 1
                if counter == self.seq_length:
                    break

        elif (len(total_frames)) >= 38:
            counter = 0
            for i in range(38):
                x = Image.open(os.path.join(path,str(i) + '.jpg')).resize((224,224))
                x = np.array(x)
                arr[:,:,:,counter] = x
                counter += 1
                if counter >= self.seq_length:
                    break
        else:
            flag = 0

        #print(arr.shape)
        return flag,arr

    def extract_feature(self, train_x):
        x_op = np.zeros((2048,40))
        for i in range(train_x.shape[3]):
            x_t = train_x[:,:,:,i]
            x_t = np.expand_dims(x_t,axis = 0)
            x = self.model.predict(x_t)
            x_op[:, i] = x

        return x_op

    def get_all_sequences(self):
        counter = 0
        y_train = []
        x_train = []

        for class_name in self.classes:  # Fight ve noFight gibi sınıflar
            directory_path = os.path.join(self.data_path, class_name)
            if class_name == 'violence':
                y = 1
            else:
                y = 0

            list_dir = os.listdir(directory_path)
            for image_file in tqdm.tqdm(list_dir):  # Resim dosyalarını iteratif olarak işleyin
                path = os.path.join(directory_path, image_file)
                flag,arr = self.get_frame_sequence(path)
                if flag == 1:
                    x_ext = self.extract_feature(arr)
                    x_train.append(x_ext)
                    counter += 1
                    y_train.append(y)

        save_file_x = os.path.join(self.saving_dir, 'data_x_ext.npy')
        save_file_y = os.path.join(self.saving_dir, 'data_y.npy')
        np.save(save_file_x, np.array(x_train))
        np.save(save_file_y, np.array(y_train))
        return x_train, y_train



def load_data():
    data_loader = DataHandler(
        data_path = 'output',
        classes = ['fight', 'noFight'],
        max_frames = 40,
        img_shape = (224, 224),
        channels = 3,
        saving_dir = 'handled_features'
    )
    x_val, y_val = data_loader.get_all_sequences()
    return x_val, y_val

if __name__ == '__main__':
    x_train, y_train = load_data()

