from tensorflow.keras.preprocessing         import image
from tensorflow.keras.applications.resnet50 import preprocess_input
import numpy as np

def pre_process(img_path = 'elephant.jpg',img_size = (224, 224) ):
    #if img_path == None:
    x = np.random.rand(img_size[0],img_size[1],img_size[2])
    #else:
    #    img = image.load_img(img_path, target_size=img_size)
    #x = image.img_to_array(img)
    
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    return x