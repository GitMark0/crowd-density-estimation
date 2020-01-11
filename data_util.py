import PIL.Image as Image
import numpy as np
import os


def get_pixels(name, path, avg=True, norm=True):
    pix = np.array(Image.open(os.path.join(path, name), 'r'))
    if avg:
        pix = avg_pool(pix)
    if norm:
        pix = normalize(pix)
    return pix

def avg_pool(pix, cluster=6, stride=6):
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


def load_image(path, index, avg=False, norm=False):
    name = 'IMG_' +  str(index) + '.jpg'
    return get_pixels(name, path, avg, norm)

def one_hot_encode(label, K):
    hot = np.zeros(K)
    hot[label] = 1
    return hot

def load_labels(set):
    with open(os.path.join(set,'labels.txt'), 'r+') as file:
        res = []
        for c in file.read().split('\n'):
            if not c:
                continue
            res.append(int(c))
        return res

def image_from_arr(arrays, gray = False, norm = True):

    type = 'L' if gray else 'RGB'
    arrays *= 255 if norm else 1
    image = Image.fromarray(arrays.astype('uint8'), type)

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
        X.append(load_image(path, i, False, True))

    return (np.array(X), np.array(load_labels(path)))


def load_data_without_labels(path, N):
    X = []
    for i in range(1, N + 1):
        X.append(load_image(path, i, False, True))

    return np.array(X)


def process_save(load_path, N, save_path):
    for i in range(1, N+1):
        img = chw_hwc(load_image(load_path, i, True, True))
        name = 'IMG_' +  str(i) + '.jpg'
        image_from_arr(img).save(os.path.join(save_path, name))


def main():
    root = 'ShanghaiTech_Crowd_Counting_Dataset'
    part_B_train = os.path.join(root,'part_B_final','train_data','images')
    part_B_test = os.path.join(root,'part_B_final','test_data','images')
    path_sets = [(part_B_train, 400), (part_B_test, 316)]

    save_path = os.path.join(root,'part_B_final','train_data','processed')
    process_save(path_sets[0][0], path_sets[0][1], save_path)
    save_path = os.path.join(root,'part_B_final','test_data','processed')
    process_save(path_sets[1][0], path_sets[1][1], save_path)

if __name__ == '__main__':
    main()
