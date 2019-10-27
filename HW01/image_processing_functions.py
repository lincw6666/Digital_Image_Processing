import numpy as np
from cv2 import cv2
import image_processing_data_structure as img_ds
import sys


def __is_image_type_valid(img):
    assert type(img)!=type(np.ndarray), '[gamma_correction] Error!! Invalid '+\
        'image type!! : '+str(type(img))


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
    return np.stack(
            [np.matmul(vec_img, kernel) for vec_img in vectorized_img], axis=-1
        ).reshape(img.shape)


def gamma_correction(img, c=1.0, gamma=1.0):
    # Check arguements.
    try:
        c = float(c)
        gamma = float(gamma)
    except:
        print('[gamma_correction] Error!! Invalid arguements\' type!!')
        sys.exit(0)

    # Main task.
    return np.uint8(c * pow(img/255.0, gamma) * 255.0)


def __check_spacial_filter_kernel_size(kernel_size, err_msg):
    try:
        kernel_size = round(kernel_size)
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


def bilateral_filter(img, kernel_size=3, sigma=1):
    kernel_size = __check_spacial_filter_kernel_size(
        kernel_size, 'bilateral_filter')
    # Build a kernel which is similar to Gaussian filter.
    half_size = kernel_size // 2
    x, y = np.mgrid[-half_size:half_size+1, -half_size:half_size+1 ]
    kernel = np.exp(-((x-y)**2)/(2.0*sigma**2)) # The only differece from
                                                # Gaussian filter.
    kernel = kernel.reshape(1, -1)
    # Apply additional weight.
    ret = __array_to_vector(img, (kernel_size, kernel_size))
    weight = np.array([
        x - np.repeat(
            [x[:, 0]], kernel_size**2, axis=0
        ).transpose() for x in ret])
    weight = np.exp(-(weight**2)/(2.0*sigma**2))
    kernel = [w*kernel for w in weight]
    kernel_sum = np.array(
        [np.sum(k, axis=-1).transpose()[:, None] for k in kernel]
    )
    kernel_sum = np.where(kernel_sum > 0, kernel_sum, 1.0)
    kernel = kernel / kernel_sum
    return np.stack(
            np.sum(ret*kernel, axis=-1), axis=-1
        ).reshape(img.shape).astype(np.uint8)


def __order_filter(img, kernel_size, order_func, func_name=''):
    kernel_size = __check_spacial_filter_kernel_size(
        kernel_size, func_name)
    return np.stack(
            order_func(
                __array_to_vector(img, (kernel_size, kernel_size)), axis=2
            ), axis=-1
        ).reshape(img.shape).astype(np.uint8)


def median_filter(img, kernel_size=3):
    return __order_filter(img, kernel_size, np.median, 'median_filter')


def max_filter(img, kernel_size=3):
    return __order_filter(img, kernel_size, np.max, 'max_filter')


def min_filter(img, kernel_size=3):
    return __order_filter(img, kernel_size, np.min, 'min_filter')


def midpoint_filter(img, kernel_size=3):
    return (
            (max_filter(img, kernel_size)+min_filter(img, kernel_size))/2.0
        ).astype(np.uint8)


def sharpen_filter(img, center_value=5):
    center_value = round(center_value)
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
    ret = ret - np.min(ret)
    return (255.0*ret/np.max(ret)).astype(np.uint8)


def histogram_equalization(img):
    hist = np.zeros(256)
    for pixel in img.flatten():
        hist[pixel] += 1
    accumulate = [hist[0]]
    for x in hist:
        accumulate.append(accumulate[-1] + x)
    accumulate = np.array(accumulate)
    new_val = (accumulate-accumulate.min()) * 255
    accumulate = (
        new_val / (accumulate.max()-accumulate.min())
    ).astype(np.uint8)
    return accumulate[img].reshape(img.shape)


# List all available functions.
func_list = [
    gamma_correction,
    box_filter,
    gaussian_filter,
    bilateral_filter,
    sharpen_filter,
    midpoint_filter,
    histogram_equalization,
    median_filter,
    max_filter,
    min_filter,
]
func_name_list = [func.__name__ for func in func_list]
# The name of parameters for each function.
func_param_name = [
    ['c', 'gamma'],     # gamma_correction
    ['kernel_size'],    # box_filter
    ['kernel_size', 'sigma'],    # gaussian_filter
    ['kernel_size', 'sigma'],    # bilateral_filter
    ['center_value'],   # sharpen_filter
    ['kernel_size'],    # midpoint_filter
    [],                 # histogram_equalization
    ['kernel_size'],    # median_filter
    ['kernel_size'],    # max_filter
    ['kernel_size'],    # min_filter
]

def func_name_to_id(func_name):
    return func_name_list.index(func_name)