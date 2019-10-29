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
                'the parameter is an integer. *****'),
            tk.Label(self, text='Active list')
        ]
        self.label[0].grid(row=0, column=2, sticky=tk.NW)
        self.label[1].grid(row=1, column=2, sticky=tk.NW)
        for i in range(2, 7):
            self.label[i].grid(row=i+1, column=2)
        self.label[7].grid(row=8, column=2, columnspan=2)
        self.label[8].grid(row=0, column=4, sticky=tk.NW)

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
        
        # [Scale] Value of parameters of each function.
        self.scale_var = [tk.DoubleVar() for _ in range(5)]
        self.scale = [
            tk.Scale(
                self, from_=0, to=15, length=400, variable=self.scale_var[i],
                resolution=0.01, orient=tk.HORIZONTAL)
            for i in range(5)
        ]
        for i in range(len(self.scale)):
            self.scale[i].grid(row=3+i, column=3, sticky=tk.NW)

        # [Button]
        self.button = [
            tk.Button(
                self, height=2, width=15, text='Add function',
                command=self.add_function),
            tk.Button(
                self, height=2, width=15, text='Remove function',
                command=self.del_function),
            tk.Button(
                self, height=2, width=15, text='Apply',
                command=self.apply_function),
            tk.Button(
                self, height=2, width=15, text='Save',
                command=self.save_parameters),
            tk.Button(
                self, height=2, width=15, text='Edit',
                command=self.edit_function)
        ]
        self.button[0].grid(row=9, column=4)
        self.button[1].grid(row=11, column=4)
        self.button[2].grid(row=10, column=2, padx=10, sticky=tk.NE)
        self.button[3].grid(row=10, column=3, padx=10, sticky=tk.NW)
        self.button[4].grid(row=10, column=4)

        # [List Box] Show the active list.
        self.listbox = tk.Listbox(self, height=20)
        self.listbox.grid(row=1, column=4, rowspan=8, sticky=tk.NW)


    def __get_image_id(self):
        try:
            return self.option_var[0].get() - 1
        except:
            return -1


    def __get_func_id(self, func_name):
        if func_name in img_f.func_name_list:
            return img_f.func_name_to_id(func_name)
        else:
            return -1


    # When we change the function, we need to update the scale in the GUI.
    #
    # @id: The id of the selected function in img_ds.func.
    def __update_scale(self, id):
        img_id = self.__get_image_id()
        assert 0 <= id <= img_ds.image[img_id].func, '[__update_scale] ' +\
            'Error!! Invalid function id!!'
        if img_id != -1:
            param = img_ds.image[img_id].func_param[id]
            for i in range(len(param)):
                self.scale[i].set(param[i])
            for i in range(len(param), len(self.scale)):
                self.scale[i].set(0.0)


    # Clear the scale when we change the image or function.
    def __clear_scale(self):
        for i in range(len(self.scale)):
            self.scale[i].set(0.0)


    # Activate while option menu select an image.
    def change_image(self, img_id):
        img_id -= 1
        img = img_ds.image[img_id].img\
            if type(img_ds.image[img_id].img) != type(None)\
                else img_ds.image[img_id].origin_img
        self.canvas_photo = ImageTk.PhotoImage(image=Image.fromarray(img))
        self.canvas.itemconfig(self.image_on_canvas, image=self.canvas_photo)
        self.__clear_scale()
        self.listbox.delete(0, 'end')
        self.listbox.insert(0, *img_ds.image[img_id].func)


    # Activate while option menu select a function.
    def change_function(self, func_name):
        func_id = self.__get_func_id(func_name)
        func_param_len = len(img_f.func_param_name[func_id])
        param_len = len(self.label) - 2
        for i in range(func_param_len):
            param_name = img_f.func_param_name[func_id][i]
            self.label[i+2]['text'] = param_name
            self.scale[i].config(**img_f.param_scale[param_name])
        if param_len-1 > func_param_len:
            for i in range(func_param_len, param_len-2):
                self.label[i+2]['text'] = '---'
                self.scale[i].config(from_=0, to=15, resolution=0.01)
        self.__clear_scale()


    # When we add a function to the active list, update its parameters.
    #
    # @img_id: The [img_id]the image.
    # @func_id: The [func_id]th function in img_f.func_list
    def __add_param(self, img_id, func_id):
        img_ds.image[img_id].func_param.append(
            [self.scale_var[i].get() 
            for i in range(len(img_f.func_param_name[func_id]))]
        )


    # Add function to the active list.
    def add_function(self):
        img_id = self.__get_image_id()
        func_name = self.option_var[1].get()
        func_id = self.__get_func_id(func_name)
        if (img_id!=-1) and (func_id!=-1):
            img_ds.image[img_id].func.append(func_name)
            self.__add_param(img_id, func_id)
            self.listbox.insert('end', func_name)


    # Get the position of the selected line in the listbox.
    def __get_listbox_pos(self):
        ret = self.listbox.curselection()
        if ret != tuple():
            return ret[0]
        else:
            return -1


    # Remove the parameters when we want to remove a function from the active
    # list.
    #
    # @img_id: The [img_id]th image.
    # @id: The [id]th function in the list.
    def __del_param(self, img_id, id):
        del img_ds.image[img_id].func[id]
        del img_ds.image[img_id].func_param[id]


    # Remove the function from the active list.
    def del_function(self):
        pos = self.__get_listbox_pos()
        if pos != -1:
            # Update func and func_param in the image.
            img_id = self.__get_image_id()
            self.__del_param(img_id, pos)
            # Update GUI.
            self.listbox.delete(pos)
            self.__clear_scale()


    # Edit the function from the active list.
    def edit_function(self):
        pos = self.__get_listbox_pos()
        if pos != -1:
            # Disable option menus, buttons and listbox.
            for i in range(len(self.option_menu)):
                self.option_menu[i].config(state=tk.DISABLED)
            for i in range(len(self.button)-1):
                self.button[i].config(state=tk.DISABLED)
            self.listbox.config(state=tk.DISABLED)
            # Update GUI.
            img_id = self.__get_image_id()
            func_id = img_f.func_name_to_id(img_ds.image[img_id].func[pos])
            param_len = len(img_ds.image[img_id].func_param[pos])
            for i in range(param_len):
                param_name = img_f.func_param_name[func_id][i]
                self.label[i+2]['text'] = param_name
                self.scale[i].config(**img_f.param_scale[param_name])
                self.scale[i].set(img_ds.image[img_id].func_param[pos][i])
            for i in range(param_len, len(self.label)-4):
                self.label[i+2]['text'] = '---'
                self.scale[i].config(from_=0, to=15, resolution=0.01)
                self.scale[i].set(0.0)
            self.button[-1]['text'] = 'Done'
            # Change button command.
            self.button[-1].config(command=self.done_edit_function)


    # Finish editting the function in the active list.
    def done_edit_function(self):
        pos = self.__get_listbox_pos()
        if pos != -1:
            # Save the parameters.
            img_id = self.__get_image_id()
            param_len = len(img_ds.image[img_id].func_param[pos])
            img_ds.image[img_id].func_param[pos] = [
                self.scale_var[i].get() for i in range(param_len)
            ]
            # Enable option menus, buttons and listbox.
            for i in range(len(self.option_menu)):
                self.option_menu[i].config(state=tk.NORMAL)
            for i in range(len(self.button)-1):
                self.button[i].config(state=tk.NORMAL)
                self.listbox.config(state=tk.NORMAL)
            # Update GUI.
            self.__clear_scale()
            func_id = img_f.func_name_to_id(self.option_var[1].get())
            param_len = len(img_f.func_param_name[func_id])
            for i in range(param_len):
                param_name = img_f.func_param_name[func_id][i]
                self.label[i+2]['text'] = param_name
                self.scale[i].config(**img_f.param_scale[param_name])
            for i in range(param_len, len(self.label)-4):
                self.label[i+2]['text'] = '---'
                self.scale[i].config(from_=0, to=15, resolution=0.01)
            self.button[-1]['text'] = 'Edit'
            # Change button command.
            self.button[-1].config(command=self.edit_function)


    # Apply all image processing function to the image.
    def apply_function(self):
        img_id = self.__get_image_id()
        if img_id != -1:
            img_ds.image[img_id].apply_processing_function()
            self.canvas_photo =\
                ImageTk.PhotoImage(
                    image=Image.fromarray(img_ds.image[img_id].img))
            self.canvas.itemconfig(
                self.image_on_canvas, image=self.canvas_photo)


    # When the result is good, you can save the parameters and output the 
    # image.
    def save_parameters(self):
        img_id = self.__get_image_id()
        if img_id != -1:
            img_ds.image[img_id].save_param()
            img_ds.image[img_id].write_image()


# Start point of the GUI app.
def start_GUI():
    global tk_frame, app
    tk_frame = tk.Tk()
    tk_frame.title('Image Processing Homework 01')
    app = Application(tk_frame)
    tk_frame.mainloop()
