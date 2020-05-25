from keras.preprocessing.image import load_img, img_to_array
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np
import os
import glob

# 入力ディレクトリを作成
input_dir = "image_input"
files = glob.glob(input_dir + '/*.jpg')

# 出力ディレクトリを作成
output_dir = "image_out"
if os.path.isdir(output_dir) == False:
    os.mkdir(output_dir)


for i, file in enumerate(files):

    img = load_img(file)
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)

    # ImageDataGeneratorの生成
    datagen = ImageDataGenerator(
        channel_shift_range=100
    )

    # 9個の画像を生成します
    g = datagen.flow(x, batch_size=1, save_to_dir=output_dir, save_prefix='img', save_format='jpg')
    for i in range(9):
        batch = g.next()
