from PIL import Image
import os, glob
import numpy as np

folders = ['folder1', 'folder2']
num_folders = len(folders)
image_size = 50

X = []
Y = []

for index, name in enumerate(folders):
    dir = './' + name
    files = glob.glob(dir + '/*.jpg')
    for i, file in enumerate(files):

        image = Image.open(file)
        image = image.convert("RGB")
        image = image.resize((image_size, image_size))
        data = np.asarray(image)
        X.append(data)
        Y.append(index)

X = np.array(X)
Y = np.array(Y)

len(X)
len(Y)
Y

xy = (X, Y)
np.save('./test.npy', xy)
