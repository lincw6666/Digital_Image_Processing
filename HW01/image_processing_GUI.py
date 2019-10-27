import tkinter as tk
from PIL import Image, ImageTk
import time
import image_processing_data_structure as img_ds
import image_processing_functions as img_f

tk_frame = None
app = None


class Application(tk.Frame):
    def __init__(self, master):
        tk.Frame.__init__(self, master)
        self.grid()
        self.create_widgets()


    # Create all the component in the GUI.
    def create_widgets(self):
        # [Canvas] Show the image.
        height, width = img_ds.image[0].origin_img.shape[:2]
        self.canvas_photo =\
            ImageTk.PhotoImage(
                image=Image.fromarray(img_ds.image[0].origin_img))
        self.canvas = tk.Canvas(self, width=width, height=height)
        self.canvas.grid(row=0, column=0, rowspan=16, sticky=tk.NW)
        self.image_on_canvas =\
            self.canvas.create_image(
                0, 0, image=self.canvas_photo, anchor=tk.NW)
        
        # [Label]
        self.label = [
            # Labels put in front of option menus.
            tk.Label(self, text='Now image: '),
            tk.Label(self, text='Image processing function: '),
            # Labels put in front of scale bars.
            tk.Label(self, text='Parameter 01'),
            tk.Label(self, text='Parameter 02'),
            tk.Label(self, text='Parameter 03'),
            tk.Label(self, text='Parameter 04'),
            tk.Label(self, text='Parameter 05'),
            tk.Label(self, text='***** [Hint] The value will be rounded if '+\
                'the parameter is an integer. *****')
        ]
        self.label[0].grid(row=0, column=2, sticky=tk.NW)
        self.label[1].grid(row=1, column=2, sticky=tk.NW)
        for i in range(2, 7):
            self.label[i].grid(row=i+1, column=2)
        self.label[7].grid(row=8, column=2, columnspan=2)

        # [Option Menu]
        # Variables start with 'option_...[]' are related to one of the option
        # menu. The ith member means:
        #   0: Choose image which we are going to process.
        #   1: Which image processing function we are using now.
        self.option_list = [
            [i for i in range(1,7)],
            [func.__name__ for func in img_f.func_list]
        ]
        self.option_var = [ tk.IntVar(), tk.StringVar() ]
        self.option_menu = [
            tk.OptionMenu(
                self, self.option_var[0], *self.option_list[0],
                command=self.change_image),
            tk.OptionMenu(
                self, self.option_var[1], *self.option_list[1],
                command=self.change_function)
        ]
        # [Option Menu] Choose image.
        self.option_var[0].set("<-- Choose image -->")
        self.option_menu[0].grid(row=0, column=3, sticky=tk.NW)
        # [Option Menu] Choose image processing function.
        self.option_var[1].set("<-- Choose function -->")
        self.option_menu[1].grid(row=1, column=3, sticky=tk.NW)

        # [Check Button] Decide that whether we use the function or not.
        self.check_button_var = tk.IntVar()
        self.check_button = tk.Checkbutton(
            self, variable=self.check_button_var, onvalue=1, offvalue=0,
            command=self.use_function)
        self.check_button["text"] = "Use this image processing function?"
        self.check_button.grid(row=2, column=2, columnspan=2, sticky=tk.NW)
        
        # [Scale] Value of parameters of each function.
        self.scale_var = [tk.DoubleVar() for _ in range(5)]
        self.scale = [
            tk.Scale(
                self, from_=0, to=10, length=400, variable=self.scale_var[i],
                resolution=0.01, orient=tk.HORIZONTAL,
                command=self.update_param)
            for i in range(5)
        ]
        for i in range(len(self.scale)):
            self.scale[i].grid(row=3+i, column=3, sticky=tk.NW)

        # [button]]
        self.button = [
            tk.Button(
                self, height=2, width=15, text='Apply',
                command=self.apply_function),
            tk.Button(
                self, height=2, width=15, text='Save',
                command=self.save_parameters)
        ]
        self.button[0].grid(row=15, column=2, padx=10, sticky=tk.NE)
        self.button[1].grid(row=15, column=3, padx=10, sticky=tk.NE)


    def __is_image_id_valid(self):
        return type(self.option_var[0].get()) == int


    def __is_func_name_valid(self, func_name):
        return func_name in img_f.func_name_list


    # When we change the function or image, we need to update the check button
    # in the GUI.
    def __update_check_button(self, func_name):
        if self.__is_image_id_valid() and self.__is_func_name_valid(func_name):
            img_id = self.option_var[0].get() - 1
            func_id = img_f.func_name_to_id(func_name)
            if img_ds.image[img_id].valid_func(func_id):
                self.check_button.select()
            else:
                self.check_button.deselect()


    # When we change the function or image, we need to update the scale in the
    # GUI.
    def __update_scale(self, func_name):
        if self.__is_image_id_valid() and self.__is_func_name_valid(func_name):
            img_id = self.option_var[0].get() - 1
            func_id = img_f.func_name_to_id(func_name)
            param_len = len(img_ds.image[img_id].func_param[func_id])
            for i in range(param_len):
                self.scale[i].set(img_ds.image[img_id].func_param[func_id][i])
            for i in range(param_len, len(self.scale)):
                self.scale[i].set(0.0)


    # Activate while option menu select an image.
    def change_image(self, img_id):
        img_id -= 1
        img = img_ds.image[img_id].img\
            if type(img_ds.image[img_id].img) != type(None)\
                else img_ds.image[img_id].origin_img
        self.canvas_photo = ImageTk.PhotoImage(image=Image.fromarray(img))
        self.canvas.itemconfig(self.image_on_canvas, image=self.canvas_photo)
        self.__update_check_button(self.option_var[1].get())
        self.__update_scale(self.option_var[1].get())


    # Activate while option menu select a function.
    def change_function(self, func_name):
        func_id = img_f.func_name_to_id(func_name)
        func_param_len = len(img_f.func_param_name[func_id])
        param_len = len(self.label) - 2
        for i in range(func_param_len):
            self.label[i+2]['text'] = img_f.func_param_name[func_id][i]
        if param_len-1 > func_param_len:
            for i in range(func_param_len, param_len-1):
                self.label[i+2]['text'] = '---'
        self.__update_check_button(func_name)
        self.__update_scale(func_name)


    # Activate while check button validates a function.
    def use_function(self):
        func_name = self.option_var[1].get()
        if self.__is_image_id_valid() and self.__is_func_name_valid(func_name):
            img_id = self.option_var[0].get() - 1
            img_ds.image[img_id].set_func_valid(
                img_f.func_name_to_id(func_name),
                is_valid=(self.check_button_var.get()==1)
            )
            self.update_param(0)


    # When we scroll the scale bar, update the parameters.
    def update_param(self, value):
        func_name = self.option_var[1].get()
        if self.__is_image_id_valid() and self.__is_func_name_valid(func_name):
            img_id = self.option_var[0].get() - 1
            func_id = img_f.func_name_to_id(func_name)
            if img_ds.image[img_id].valid_func(func_id):
                for i in range(len(img_ds.image[img_id].func_param[func_id])):
                    img_ds.image[img_id].func_param[func_id][i] =\
                        self.scale_var[i].get()


    # Apply all image processing function to the image.
    def apply_function(self):
        if self.__is_image_id_valid():
            img_id = self.option_var[0].get() - 1
            img_ds.image[img_id].apply_processing_function()
            self.canvas_photo =\
                ImageTk.PhotoImage(
                    image=Image.fromarray(img_ds.image[img_id].img))
            self.canvas.itemconfig(
                self.image_on_canvas, image=self.canvas_photo)


    # When the result is good, you can save the parameters and output the 
    # image.
    def save_parameters(self):
        if self.__is_image_id_valid():
            img_id = self.option_var[0].get() - 1
            img_ds.image[img_id].save_param()
            img_ds.image[img_id].write_image()


# Start point of the GUI app.
def start_GUI():
    global tk_frame, app
    tk_frame = tk.Tk()
    tk_frame.title('Image Processing Homework 01')
    app = Application(tk_frame)
    tk_frame.mainloop()
