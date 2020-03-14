import cv2
from simple_jpeg import Compression
from evaluation import RMS, SNR

if __name__ == '__main__':
    img_path = 'images/lena.bmp'
    origin_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    methods = ['WHT','DCT', 'DFT']
    block_size = [4, 4, 4]
    N_K = [512, 512, 512]
    for i in range(len(methods)):
        compress = Compression(
            img_path, block_size[i],
            transform_method=methods[i],
            quantization_method='Total N',
            N_K=N_K[i])
        
        # Apply compression and reconstruct it.
        compress.compress()
        compress.reconstruct()
        cv2.imwrite(f'outputs/{methods[i]}.bmp', compress.img)

        # Evaluate by RMS and SNR.
        rms = RMS(compress.img, origin_img)
        snr = SNR(compress.img, origin_img)
        print(f'RMS: {rms}, SNR(dB): {snr}')
