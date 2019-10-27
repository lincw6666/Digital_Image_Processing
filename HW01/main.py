import image_processing_data_structure as img_ds
import image_processing_functions as img_f
import image_processing_GUI as img_GUI

# Put images under 'image_dir'. Then load images.
img_ds.load_image(dir='./images/')

img_GUI.start_GUI()
