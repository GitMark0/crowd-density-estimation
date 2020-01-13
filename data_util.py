import PIL.Image as Image
import numpy as np
import os
from matplotlib import pyplot as plt


def get_pixels(image, avg=True, norm=True):
    """
        Returns a numpy array of pixels given a PIL Image object.
        If avg is set, average pools the pixels, if norm is set normalizes the
        pixels to 0-1 scale.
    """
    pix = np.array(image)
    if avg:
        pix = avg_pool(pix)
    if norm:
        pix = normalize(pix)
    return pix


def avg_pool(pix, cluster=6, stride=6):
    """
        Average pools a numpy array according to cluster and stride values.
        Output array size will be equal:
            (height-cluster)/stride + 1
            (width-cluster)/stride + 1
    """
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
    """
        Normalize pixel values by dividing with max value which equals 255 for
        images.
    """
    return pix/255


def load_image(image, avg=False, norm=False):
    return get_pixels(image, avg, norm)


def image_from_arr(arrays, gray = False, norm = True):
    """
        Returns a PIL image object given a numpy array of images.
        If gray is set image should have only 1 channel, otherwise 3.
        Set norm to true if pixels are normalized to 0-1 scale.
    """
    type = 'L' if gray else 'RGB'
    arrays *= 255 if norm else 1
    image = Image.fromarray(arrays.astype('uint8'), type)

    return image


def chw_hwc(arrays):
    """
        Change a numpy matrix of pixels from channel-height-width to
        height-width-channel format.
        (channel is a RGB component array of a pixel)
    """
    red_arr,green_arr,blue_arr = arrays[0],arrays[1],arrays[2]
    image_arr = np.empty((red_arr.shape[0], red_arr.shape[1], 3))
    for i in range(0, red_arr.shape[0]):
        for j in range(0, red_arr.shape[1]):
            image_arr[i,j] = \
                np.array([red_arr[i,j], green_arr[i,j], blue_arr[i,j]])
    return image_arr


def prepare_image(image, target_dim):
    """
        Returns image scaled to given target dimension.
        If target dimensions are horizontal (width > height), rotates the image
        90 degrees.
    """
    original_size = image.size
    image_horiz = image.size[0] > image.size[1]
    target_horiz = target_dim[0] > target_dim[1]
    rotate = image_horiz != target_horiz
    if rotate:
        image = image.rotate(90)
    return ((original_size, rotate), image.resize((target_dim[1], target_dim[0])))


def load_example(path, target_dim, norm=True):
    """
        Loads an example, combining prepare_image and get_pixels functions.
    """
    info, image = prepare_image(Image.open(os.path.join(path), 'r'), target_dim)
    return (info, get_pixels(image, False, norm))


def show_prediction_as_image(image, size, rotate):
    """
        Plots a numpy array of predicted heatmap pixel values.
        Reshapes and rotates the heatmap to fit original image dimensions.
    """
    plt.figure(figsize=(12,12))
    image = image_from_arr(image, gray=False, norm=True).convert('LA')
    if rotate:
        image = image.rotate(270)
    image = image.resize(size)
    image = np.array(image)[:, :, 0]

    plt.imshow(image, cmap='hot', vmin=21, vmax=120)
