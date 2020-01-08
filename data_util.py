import PIL.Image as Image
import numpy as np
import os


def get_pixels(name, path, avg = True, norm = True):
    pix = np.array(Image.open(os.path.join(path, name), 'r'))
    if avg:
        pix = avg_pool(pix)
    if norm:
        pix = normalize(pix)
    return pix

def avg_pool(pix, cluster = 5, stride = 4):
    input_shape = pix.shape
    output_shape = (int((input_shape[0] - cluster)/stride) + 1,\
                int((input_shape[1] - cluster)/stride) + 1)
    output = []

    for channel in range(pix.shape[2]):
        output.append(np.empty(output_shape))

        for i in range(0, output_shape[0]):
            for j in range(0, output_shape[1]):

                sum = 0

                for a in range(0, cluster):
                    for b in range(0, cluster):
                        sum += pix[i * stride + a, j * stride + b, channel]

                output[channel][i,j] = sum/(cluster * cluster)

    return np.array(output)

def normalize(pix):
    return pix/255


def load_image(path, index):
    name = 'IMG_' +  str(index) + '.jpg'
    return get_pixels(name, path)

def load_labels(set):
    with open(os.path.join(set,'labels.txt'), 'r+') as file:
        res = []
        for c in file.read().split('\n'):
            if not c:
                continue
            res.append(int(c))
        return res

def image_from_arr(arrays, gray = False):

    image_arr = None
    type = None

    if gray :
        type = 'L'
        image_arr = arrays[0]*255
    else :
        type = 'RGB'
        red_arr,green_arr,blue_arr = arrays[0]*255,arrays[1]*255,arrays[2]*255
        image_arr = np.empty((red_arr.shape[0], red_arr.shape[1], 3))
        for i in range(0, red_arr.shape[0]):
            for j in range(0, red_arr.shape[1]):
                image_arr[i,j] = \
                    np.array([red_arr[i,j], green_arr[i,j], blue_arr[i,j]])
    image = Image.fromarray(image_arr.astype('uint8'), type)

    return image

def chw_hwc(arrays):
    red_arr,green_arr,blue_arr = arrays[0],arrays[1],arrays[2]
    image_arr = np.empty((red_arr.shape[0], red_arr.shape[1], 3))
    for i in range(0, red_arr.shape[0]):
        for j in range(0, red_arr.shape[1]):
            image_arr[i,j] = \
                np.array([red_arr[i,j], green_arr[i,j], blue_arr[i,j]])
    return image_arr

def load_data(path, N):
    load_labels(path)
    X = []
    for i in range(1, N+1):
        X.append(chw_hwc(load_image(path, i)))
        print(X[-1].shape)

    return (np.array(X), np.array(load_labels(path)))
