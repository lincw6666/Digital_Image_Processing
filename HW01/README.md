Image Processing Homework 01
===

# Introduction

There are 6 images given, 5 color images and 1 CT image. The goal is to produce nice images by some image processing techniques as best as we can.
\* Remark: I'm sorry that I can't provide the images due to the copyright.

# Run the code

- Make sure that you have already installed **opencv on python**
- Put all these 4 files under the same directory.
- Put all the images (only accept .bmp format) under `./images/`. And rename them as p1im*.bmp, where * is 1 ~ 6.
- Create `./parameters/`. All the parameters for each function of each image are stored at here.
- Create `./results/`. The output images are stored at here.
- Run the code:
    ```python=
    $ python3.7 main.py
    ```
    You can change `python3.7` to other python version.

# GUI

![](https://i.imgur.com/LiT9KWF.jpg)
This is the overall interface. I'll introduce all its functions in the following section.
- `Now image`: The image you want to process now. The images are come from `./images/`.
- `Image Processing function`: Choose the function you want to apply to the image.
- `Scale bars`: The parameters for the function. Please remind that **kernel size MUST be an odd number**. You might choose an even number, but it's invalid and it'll raise an exception then shutdown the APP.
- `Active list`: The functions you've applied on the images. Please remind that **the order does matter to the non-linear functions**.
    - `Add function`: Add function to the active list.
    - `Edit`: If you want to modify the parameters of the function in the active list. You need to select the function in the list, then press the `Edit` button. Now, you'll see the figure below.
    ![](https://i.imgur.com/U6sLwwx.jpg)
    After finishing modifying the function, press the `Done` button, then you will come back to the origin GUI. Please remind that you can't press any button other than `Done` button. You also can't use the option menus.
- `Apply`: It'll apply all funtions in the active list to the image. The output will be shown on the left.
- `Save`: Save the output image to `./results/` and save the parameters for each function to `./parameters/`.
