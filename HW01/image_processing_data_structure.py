import numpy as np
from cv2 import cv2
import image_processing_functions as img_f
import pandas as pd
import sys

image = None


# It stores the image itself and all the parameters for the image processing
# functions.
class My_Image:
    # @origin_img: Store the np.ndarray of input image.
    # @img: Store the modified image.
    # @is_func_valid: Show which image processing function is applying now.
    #       It's a bitmap.
    # @func_param: Store the parameters for each function.
    def __init__(self, path):
        self.origin_img = None
        self.img = None
        self.img_name = None
        self.is_func_valid = int(0)
        self.func_param = [
            [0.0 for _ in img_f.func_param_name[i]]\
                for i in range(len(img_f.func_param_name))]
        self.load_image(path)
    
    # @path: It should start with './'. EX: './images/'.
    def load_image(self, path):
        assert type(path)!=type(str), '[Image:load_image] \'path\' need to be'+\
            ' a string!!'
        image = cv2.imread(path)
        # Check whether the image exists.
        if type(image) == type(None):
            # The root directory for the vscode environment locates at
            # 'D:/Course/Graduate_1/'. This try block makes it runnable on
            # vscode.
            path = path[0] + '/Image_Processing/Homework/HW01' + path[1:]
            image = cv2.imread(path)
            if type(image) == type(None):
                print('[Image:load_image] Error!! File not found!! :', path)
                sys.exit(0)
        self.origin_img = image
        self.img_name = path

    def write_image(self):
        path = self.img_name.split('/')
        path[-1] = path[-1].replace('p', 'P', 1)
        path[-1] = path[-1][:5] + '_0856030' + path[-1][5:]
        path[-2] = 'results'
        path = '/'.join(path)
        try:
            if type(self.img) != type(None):
                cv2.imwrite(path, self.img)
            else:
                cv2.imwrite(path, self.origin_img)
        except:
            print('[Image:write_image]',\
                'Error!! Can\'t save image to file!! :', path)
    
    # If you want to use the img_f.func_list[func_id] function, you need to set
    # the corresponding bit in is_func_valid to 1. If you don't want to use it,
    # set to 0. 'set_func_valid' helps you to do so.
    # 
    # @func_id: The [func_id]th function in img_f.func_list.
    # @is_valid: Default to True. True if the function is valid, vice versa. 
    def set_func_valid(self, func_id, is_valid=True):
        assert type(func_id)!=type(int), '[Image:set_func_valid] \'func_id\''+\
            ' needs to be an integer!!'
        assert type(is_valid)!=type(bool), '[Image:set_func_valid] '+\
            '\'is_valid\' needs to be an integer!!'
        if is_valid:
            self.is_func_valid = self.is_func_valid | (1<<func_id)
        else:
            self.is_func_valid = self.is_func_valid & ~(1<<func_id)


    # Check whether the [func_id]th function is valid.
    # @func_id: The [func_id]th function.
    def valid_func(self, func_id):
        return (self.is_func_valid&(1<<func_id)) != 0


    # Save the parameters for each function to a file. One line contains
    # parameters for exactly one function, with comma seperates them.
    # 
    # Example:
    # The following shows that (1, 2, 3) for func_1, (4, 5, 6) for func_2.
    #     1,2,3
    #     4,5,6
    #
    # @path: Path to the file. Please remind that it must be a .csv file.
    def save_param(self):
        path = self.img_name.split('/')
        path[-1] = path[-1].replace('bmp', 'csv')
        path[-2] = 'parameters'
        path = '/'.join(path)
        
        try:
            df = pd.DataFrame({
                'is_func_valid': self.is_func_valid,
                'func_param': self.func_param
                })
            df.to_csv(path, index=False)
        except:
            print('[Image:save_param]',\
                'Error!! Can\'t save parameters to file!! :', path)

    # Apply all the valid image processing functions on the image.
    def apply_processing_function(self):
        self.img = self.origin_img
        for i in range(len(img_f.func_list)):
            if self.valid_func(i):
                self.img = img_f.base_func_wrapper(
                    img_f.func_list[i],
                    self.img,
                    **dict(zip(img_f.func_param_name[i], self.func_param[i])))


def load_image(dir='./images/'):
    global image
    image_name = [dir+'p1im'+str(i)+'.bmp' for i in range(1, 7)]
    image = [My_Image(img_name) for img_name in image_name]
