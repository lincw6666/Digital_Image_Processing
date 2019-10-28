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
        self.func = []
        self.func_param = []
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


    # Save the parameters for each function to a file. One line contains
    # parameters for exactly one function, with comma seperates them.
    # 
    # Example:
    # The following shows that (1, 2, 3) for func_1, (4, 5, 6) for func_2.
    #     1,2,3
    #     4,5,6
    #
    def save_param(self):
        path = self.img_name.split('/')
        path[-1] = path[-1].replace('bmp', 'csv')
        path[-2] = 'parameters'
        path = '/'.join(path)
        
        try:
            df = pd.DataFrame({
                'func_name': self.func,
                'func_param': self.func_param
                })
            df.to_csv(path, index=False)
        except:
            print('[Image:save_param]',\
                'Error!! Can\'t save parameters to file!! :', path)

    # Apply all the valid image processing functions on the image.
    def apply_processing_function(self):
        self.img = self.origin_img
        for i in range(len(self.func)):
            func_id = img_f.func_name_to_id(self.func[i])
            self.img = img_f.base_func_wrapper(
                img_f.func_list[func_id], self.img,
                **dict(
                    zip(
                        img_f.func_param_name[func_id], self.func_param[i]
                    )
                )
            )


def load_image(dir='./images/'):
    global image
    image_name = [dir+'p1im'+str(i)+'.bmp' for i in range(1, 7)]
    image = [My_Image(img_name) for img_name in image_name]
