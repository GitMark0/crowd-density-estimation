import shutil
import os
import random

img_indexes = random.sample(range(550), 110)
rootdir = '/media/dabar/C0CA6608CA65FB54/PycharmProjects/Raspoznavanje_gustoca/Dataset/train/sparse/'
i = 0

for subdir, dirs, files in os.walk(rootdir):
    for file in files:
        filepath = subdir + os.sep + file
        if i in img_indexes:
            print(filepath)
            shutil.move(filepath,
                        '/media/dabar/C0CA6608CA65FB54/PycharmProjects/Raspoznavanje_gustoca/Dataset/test/sparse/')

        i += 1
