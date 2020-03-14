import numpy as np


def RMS(img, ref_img):
    assert img.shape == ref_img.shape, \
        '[RMS] The shape of img and ref_img must be consistant!'
    total_pixels = img.shape[0] * img.shape[1]
    return np.sqrt(np.sum((img-ref_img)**2)/total_pixels)


def SNR(img, ref_img):
    assert img.shape == ref_img.shape, \
        '[RMS] The shape of img and ref_img must be consistant!'
    return 10 * np.log10(np.sum(img**2)/np.sum((img-ref_img)**2))
