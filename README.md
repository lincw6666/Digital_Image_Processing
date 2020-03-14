Digital Image Processing
===
2019 Fall - Course - NCTU (Graduate) Digital Image Processing

# Homework 01
## Introduction
Enhance the supplied images using techniques of contrast adjustment, noise reduction, color correction, and so on. There are 6 images supplied (5 regular photos, and one CT).

⚠ Implement image processing methods with the restriction on toolbox/library usage.

- Origin images
  ![](https://i.imgur.com/DM6c1B3.jpg)

## GUI
Support a GUI for parameter adjustment.

![](https://i.imgur.com/CoIY4Ls.jpg)

- Support editting the parameter of the function in the activate list. You can only adjust the parameter before you press the "Done" button.

  ![](https://i.imgur.com/JILIOiN.jpg)

## Implemented Methods

- [x] Gamma correction
- [x] Histogram equalization
- [x] Box filter
- [x] Gaussian filter
- [x] Bilateral filter
- [x] Sharpen filter
- [x] Median filter
- [x] Max Filter
- [x] Min filter
- [x] Midpoint filter
- [x] HSV adjustment

## Results
Please refer to my report for details.

![](https://i.imgur.com/1Q5cKYw.jpg)

# Homework 03
## Introduction
Implement lossy image compression and study the effects of various methods and parameter choices. However, the complete JPEG standard includes several stages and options. With our time limit, we'll focus on transform coding and quantization.

## Usage
1. Install *OpenCV*.
2. Import packages.
  ```python=
  import cv2
  from simple_jpeg import Compression
  from evaluation import RMS, SNR
  ```
3. Create a *Compression* object.
  - `img_path`: Path to your input image.
  - `block_size`: The block size for compression.
  - `transform_method`: DFT, DCT, or WHT.
  - `quantization_method`: K first, K largest, Total N.
    - K first: Use first K coefficients for compression.
    - K largest: Use K largest coefficients for compression.
    - Total N: Total N bits for coefficients. How many bits for a specific coefficient depend on its variances over all blocks. Assume that the variance of a coefficient is <img src="http://latex.codecogs.com/gif.latex?v_i" />, we define <img src="http://latex.codecogs.com/gif.latex?q_i = log_2(v_i)" />. Then the number of bits for that coefficient is <img src="http://latex.codecogs.com/gif.latex?n_i = round(\frac{N*q_i}{\sum_i q_i})" />.
  - `N_K`: N or K for quantization method.
  ```python=
  compress = Compression(img_path, block_size,
                         transform_method=method,
                         quantization_method=q_method,
                         N_K=N_K)
  ```
4. Compress the image.
  ```python=
  compress.compress()
  ```
5. Reconstruct the image.
  ```python=
  compress.reconstruct()
  ```
6. Evaluate the RMS and SNR.
  ```python=
  rms = RMS(compress.img, origin_img)
  snr = SNR(compress.img, origin_img)
  ```

## Results
Please refer to my report for details.
