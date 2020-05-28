# CNNの構築
import sys
import keras
from keras.utils import np_utils
from keras.models import Sequential, load_model
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Dense, Dropout, Activation, Flatten
from PIL import Image
import numpy as np

image_size = 32
classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
'dog', 'frog', 'horse', 'ship', 'truck']
num_classes = len(classes)

def built_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same',input_shape=(32, 32, 3)))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10))
    model.add(Activation('softmax'))

    # コンパイル
    model.compile(loss='categorical_crossentropy',optimizer='SGD',metrics=['accuracy'])
    #モデルロード
    model = load_model('./CIFAR-10.h5')

    return model


def main():
    image = Image.open(sys.argv[1])
    image = image.convert('RGB')
    image = image.resize((image_size, image_size))
    data = np.asarray(image) / 255
    X = []
    X.append(data)
    X = np.array(X)
    model = built_model()

    result = model.predict([X])[0]
    predicted = result.argmax()
    percentage = int(result[predicted] * 100)
    print('{0}({1} %)'.format(classes[predicted], percentage))

if __name__ == '__main__':
    main()
