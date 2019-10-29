import numpy as np
from cv2 import cv2
import sys


def __is_image_type_valid(img):
    assert type(img)!=type(np.ndarray), '[gamma_correction] Error!! Invalid '+\
        'image type!! : '+str(type(img))


def __image_reshape(img, shape):
    __is_image_type_valid(img)
    assert img.shape[2] == 3, '[__image_reshape] Error!! Only accept 3 '+\
        'channels image!!'
    h, w, c = shape
    assert c == 3, '[__image_reshape] Error!! Only reshape to 3 channels '+\
        'image!!'
    assert img.shape[0] == h*w, '[__image_reshape] Error!! Unmatch image '+\
        'and reshape image size!!'
    return np.array([img[w*i:w*(i+1), 0, :] for i in range(h)])


# We apply all the function through this wrapper.
def base_func_wrapper(func, img, **kwargs):
    __is_image_type_valid(img)
    if not callable(func):
        print('[base_func_wrapper] Error!! Bad function!!')
        sys.exit(0)
    else:
        #try:
        return func(img, **kwargs)
        #except:
        #    sys.exit(0)
    return None


def __normalize(img):
    return np.uint8(img / 255.0)


def __array_to_vector(img, kernel_size):
    k_rows, k_cols = kernel_size
    padx, pady = k_rows//2, k_cols//2
    rows, cols, channels = img.shape
    ret = np.zeros((rows+2*padx, cols+2*pady, channels))
    ret[padx:rows+padx, pady:cols+pady, :] = img
    row_len = cols + 2*pady
    row_channel_len = row_len * channels
    start_idx = np.array([
        [j*channels+(row_channel_len)*i + k\
            for i in range(rows-k_rows+1+2*padx)\
                for j in range(cols-k_cols+1+2*pady)]\
                    for k in range(channels)])
    grid = np.array(
        np.tile([j*channels+(row_channel_len)*i\
            for i in range(k_rows) for j in range(k_cols)], (channels, 1)))
    to_take = start_idx[:, :, None] + grid[:, None, :]
    return [ret.take(to_take[i]) for i in range(channels)]


def __convolution(img, kernel):
    # Check the kernel's properties.
    assert type(kernel)!=type(np.ndarray), '[convolution] Error!! Invalid '+\
        'kernel type!!'
    
    # Main task.
    vectorized_img = __array_to_vector(img, kernel.shape)
    kernel = kernel.reshape(-1, 1)
    return __image_reshape(
        np.stack(
            [np.matmul(vec_img, kernel) for vec_img in vectorized_img], axis=-1
        ),
        img.shape)


def gamma_correction(img, c=1.0, gamma=1.0):
    # # Check arguements.
    # try:
    #     c = float(c)
    #     gamma = float(gamma)
    # except:
    #     print('[gamma_correction] Error!! Invalid arguements\' type!!')
    #     sys.exit(0)

    # Main task.
    return np.uint8(c * pow(img/255.0, gamma) * 255.0)


def __check_spacial_filter_kernel_size(kernel_size, err_msg):
    try:
        kernel_size = int(kernel_size)
        assert kernel_size%2 != 0
        return kernel_size
    except:
        print('['+err_msg+'] Error!! Invalid kernel size!! :', kernel_size)
        sys.exit(0)
    return None


def box_filter(img, kernel_size=3):
    kernel_size = __check_spacial_filter_kernel_size(kernel_size, 'box_filter')
    kernel = np.ones((kernel_size, kernel_size)) / (kernel_size**2)
    return __convolution(img, kernel).astype(np.uint8)


def gaussian_filter(img, kernel_size=3, sigma=1):
    kernel_size = __check_spacial_filter_kernel_size(
        kernel_size, 'gaussian_filter')
    half_size = kernel_size // 2
    x, y = np.mgrid[-half_size:half_size+1, -half_size:half_size+1 ]
    kernel = np.exp(-(x**2+y**2)/(2.0*sigma**2))
    kernel = kernel / kernel.sum()
    return __convolution(img, kernel).astype(np.uint8)


def bilateral_filter(img, kernel_size=3, sigma1=1, sigma2=0.1):
    kernel_size = __check_spacial_filter_kernel_size(
        kernel_size, 'bilateral_filter')
    # Build a kernel which is similar to Gaussian filter.
    half_size = kernel_size // 2
    x, y = np.mgrid[-half_size:half_size+1, -half_size:half_size+1 ]
    kernel = -(x**2+y**2)/(2.0*sigma1**2)
    kernel = kernel.reshape(1, -1)
    # Apply additional weight.
    vectorized_img = __array_to_vector(img, (kernel_size, kernel_size))
    weight = np.array([
        x - np.repeat(
            [x[:, half_size]], kernel_size**2, axis=0
        ).transpose() for x in vectorized_img])
    kernel = np.exp(kernel - (weight**2)/(2.0*sigma2**2))
    kernel_sum = np.array(
        [np.sum(k, axis=-1).transpose()[:, None] for k in kernel]
    )
    kernel_sum = np.where(kernel_sum > 0, kernel_sum, 1.0)
    kernel = kernel / kernel_sum
    return __image_reshape(
        np.stack(
            [x for x in np.sum(vectorized_img*kernel, axis=-1)], axis=-1
        )[:, None, :],
        img.shape).astype(np.uint8)    
    


def __order_filter(img, kernel_size, order_func, func_name=''):
    kernel_size = __check_spacial_filter_kernel_size(
        kernel_size, func_name)
    return __image_reshape(
        np.stack(
            order_func(
                __array_to_vector(img, (kernel_size, kernel_size)), axis=2
            ), axis=-1
        )[:, None, :],
        img.shape).astype(np.uint8)


def median_filter(img, kernel_size=3):
    return __order_filter(img, kernel_size, np.median, 'median_filter')


def max_filter(img, kernel_size=3):
    return __order_filter(img, kernel_size, np.max, 'max_filter')


def min_filter(img, kernel_size=3):
    return __order_filter(img, kernel_size, np.min, 'min_filter')


def midpoint_filter(img, kernel_size=3):
    kernel_size = __check_spacial_filter_kernel_size(
        kernel_size, 'midpoint_filter')
    lab_img = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
    min_l = np.min(
        __array_to_vector(
            lab_img[:, :, 0][:, :, None], (kernel_size, kernel_size)), axis=2)
    max_l = np.max(
        __array_to_vector(
            lab_img[:, :, 0][:, :, None], (kernel_size, kernel_size)), axis=2)
    img_l = (min_l+max_l) / 2.0
    lab_img[:, :, 0] = np.array(
        [img_l[0, img.shape[1]*i:img.shape[1]*(i+1)]
        for i in range(img.shape[0])])
    return cv2.cvtColor(lab_img, cv2.COLOR_Lab2BGR).astype(np.uint8)


def sharpen_filter(img, center_value=5):
    center_value = int(center_value)
    assert (center_value==5) or (center_value==9),\
        '[sharpen_filter] Error!! Invalid center value!! : '+\
            str(center_value)
    
    if center_value == 5:
        kernel = np.array(
            [[0, -1, 0],
             [-1, 5, -1],
             [0, -1, 0]])
    else:
        kernel = np.array(
            [[-1, -1, -1],
             [-1, 9, -1,],
             [-1, -1, -1]])
    ret = __convolution(img, kernel)
    return np.clip(ret, 0, 255).astype(np.uint8)


def histogram_equalization(img, min_val, max_val):
    hist = np.zeros(256)
    for pixel in img.flatten():
        hist[pixel] += 1
    accumulate = [hist[0]]
    for x in hist:
        accumulate.append(accumulate[-1] + x)
    accumulate = np.array(accumulate)
    new_val = (accumulate-accumulate.min()) * (max_val-min_val) + min_val
    accumulate = (
        new_val / (accumulate.max()-accumulate.min())
    ).astype(np.uint8)
    return accumulate[img]


def HSV_adjust(img, H=0, S=0, V=0):
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv_img = np.clip(hsv_img+np.array([H, S, V]), 0, 255)
    return cv2.cvtColor(hsv_img.astype(np.uint8), cv2.COLOR_HSV2BGR)


# List all available functions.
func_list = [
    gamma_correction,
    histogram_equalization,
    box_filter,
    gaussian_filter,
    bilateral_filter,
    sharpen_filter,
    median_filter,
    max_filter,
    min_filter,
    midpoint_filter,
    HSV_adjust,
]
func_name_list = [func.__name__ for func in func_list]
# The name of parameters for each function.
func_param_name = [
    ['c', 'gamma'],     # gamma_correction
    ['min_val', 'max_val'],                 # histogram_equalization
    ['kernel_size'],    # box_filter
    ['kernel_size', 'sigma'],    # gaussian_filter
    ['kernel_size', 'sigma1', 'sigma2'],    # bilateral_filter
    ['center_value'],   # sharpen_filter
    ['kernel_size'],    # median_filter
    ['kernel_size'],    # max_filter
    ['kernel_size'],    # min_filter
    ['kernel_size'],    # midpoint_filter
    ['H', 'S', 'V'],    # HSV_adjust
]
param_scale = {
    'c': {'from_': 0, 'to': 2, 'resolution': 0.01},
    'gamma': {'from_': 0, 'to': 10, 'resolution': 0.01},
    'min_val': {'from_': 0, 'to': 100, 'resolution': 1.0},
    'max_val': {'from_': 155, 'to': 255, 'resolution': 1.0},
    'kernel_size': {'from_': 3, 'to': 15, 'resolution': 1.0},
    'sigma': {'from_': 1, 'to': 5, 'resolution': 1.0},
    'sigma1': {'from_': 1, 'to': 100, 'resolution': 1.0},
    'sigma2': {'from_': 1, 'to': 100, 'resolution': 1.0},
    'center_value': {'from_': 5, 'to': 9, 'resolution': 1.0},
    'H': {'from_': -100, 'to': 100, 'resolution': 1.0},
    'S': {'from_': -100, 'to': 100, 'resolution': 1.0},
    'V': {'from_': -100, 'to': 100, 'resolution': 1.0},
}

def func_name_to_id(func_name):
    return func_name_list.index(func_name)