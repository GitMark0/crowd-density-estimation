import os
path = '/media/dabar/C0CA6608CA65FB54/PycharmProjects/Raspoznavanje_gustoca/Dataset/test/densse/'
files = os.listdir(path)


for index, file in enumerate(files):
    filename = os.path.join(path, file)
    if filename.endswith('.jpg') or filename.endswith('.png') or filename.endswith('.jpeg'):
        extension = file[file.index('.'):-1] + file[-1]
        os.rename(filename, os.path.join(path, ''.join([str(index), '_ae'+extension])))