import keras
backend = keras.backend
layers  = keras.layers 
models  = keras.models 
utils   = keras.utils
from keras.layers        import Input

from common.generator.inception_generator import inception_n_layers
from common.generator.inception_generator import inception_1_layer
from common.generator.inception_generator import InceptionGenerator

from common.generator.conv_generator      import conv_filter_wise
from common.generator.conv_generator      import WidhtGenerator 

from common.generator.conv_generator      import conv_depth_wise
from common.generator.conv_generator      import DeepGenerator

from common.generator.resnet_generator    import resnet_1_layer_filter_wise
from common.generator.resnet_generator    import ResWidhtGenerator
   
from common.generator.resnet_generator    import ResDephtGenerator
from common.generator.resnet_generator    import resnet_n_layers

import itertools



class Generator:
    def __init__(self) -> None:
        self.constructor = {}

    def run(self):
        
        self.constructor.update(self.conv_1L()  )
        self.constructor.update(self.conv_N()   ) 
        self.constructor.update(self.res_1L()   )  
        self.constructor.update(self.res_N()    )   
        self.constructor.update(self.inception())  

        return self.constructor

        
    def conv_1L(self):

        filters     = [1,2,3,8,16,32,64,128,256]
        kernel_size = [1,2,3,4,5,6,7]
        strides     = [(1, 1),(2, 2),(3, 3),(4, 4)]
        include_top = [True,False]
        padding     = ["valid","same"]

        params = list(itertools.product(filters,kernel_size,strides,include_top,padding))

        constructor = {"CONV_1L_"+str(p[:-1]).replace(',','_')
                                        .replace(' ','')
                                        .replace('','')
                                        .replace(')','')
                                        .replace('(','') + '_' +
                                        p[-1]:
                                        WidhtGenerator(conv_filter_wise,p).Constructor for p in params}
        return constructor 

    
    def conv_N(self):

        deep        = [3,8,16,32]
        kernel_size = [1,2,3]
        strides     = [(1, 1),(2, 2)]
        include_top = [True,False]
        padding     = ["valid","same"]
        pyramidal   = ["increasing","decreasing","retangular"]

        params = list(itertools.product(deep,kernel_size,strides,include_top,padding,pyramidal))



        constructor = {"CONV_N"+str(p[:-2]).replace(',','_')
                                        .replace(' ','')
                                        .replace('','')
                                        .replace(')','')
                                        .replace('(','') + '_' + p[-2]+'_'+p[-1]
                                        :DeepGenerator(conv_depth_wise,p).Constructor for p in params}
        return constructor

    def res_1L(self):

        filters     = [1,2,3,8,16,32]
        kernel_size = [1,2,3,4,5]
        include_top = [True,False]
        stride      = [1,2]

        params = list(itertools.product(filters,kernel_size,include_top,stride))


        constructor = {"RES_L1"+str(p).replace(',','_')
                                        .replace(' ','')
                                        .replace('','')
                                        .replace(')','')
                                        .replace('(','')
                                        :ResWidhtGenerator(resnet_1_layer_filter_wise,p).Constructor for p in params}

        return constructor

    def res_N(self):

        deep         = [2,3,4]
        pyramidal    = ["increasing","decreasing","retangular"]
        kernel_size  = [1,2,3]
        stride       = [1,2] 
        include_top  = [True,False]

        params = list(itertools.product(deep,pyramidal,kernel_size,stride,include_top))


        constructor = {"RES_N"+str(p).replace(',','_')
                                        .replace(' ','')
                                        .replace('','')
                                        .replace(')','')
                                        .replace('(','')
                                        :ResDephtGenerator(resnet_n_layers,p).Constructor for p in params}
        return constructor

    def inception(self):
        
        shape_1x1    = [256,128,64,32,16] 
        shape_5x5    = [256,128,64,32,16]
        shape_3x3dbl = [384,192,96,48,24] 
        shape_pool   = [128,64 ,32,16,8 ]

        constructor = {"INCEPTION_"+str(i):InceptionGenerator(inception_n_layers,
                                                    shape_1x1[-i:],
                                                    shape_5x5[-i:],
                                                    shape_3x3dbl[-i:],
                                                    shape_pool[-i:],
                                                    False).Constructor for i in range(4)}

        constructor_top = {"INCEPTION_"+str(i)+"_TOP":InceptionGenerator(inception_n_layers,
                                                    shape_1x1[-i:],
                                                    shape_5x5[-i:],
                                                    shape_3x3dbl[-i:],
                                                    shape_pool[-i:],
                                                    True).Constructor for i in range(4)}

        constructor.update(constructor_top)

        return constructor


#from keras.preprocessing         import image
#from keras.applications.resnet50 import preprocess_input
#import numpy as np
#
#def pre_process(img_path = 'elephant.jpg',img_size = (224, 224) ):
#    #if img_path == None:
#    x = np.random.rand(img_size[0],img_size[1],img_size[2])
#    #else:
#    #    img = image.load_img(img_path, target_size=img_size)
#    #x = image.img_to_array(img)
#    
#    x = np.expand_dims(x, axis=0)
#    x = preprocess_input(x)
#
#    return x
 

 

#print(constructors)
#print(len(constructors))
#
#for model_name in constructors:
#    print(model_name)
#    print(constructors[model_name])
#    model = constructors[model_name](weights=None,input_tensor=Input(shape=(128,128,3)),classes = 1)
#    x     = pre_process(img_size=(128,128,3) ) 
#    preds = model.predict(x)
#    print('model_name:',model_name,'shape:',preds.shape)



'''
resnet_1_layer_filter_wise
resnet_1_layer_filter_wise_top
resnet_n_layers
resnet_n_layers_top
inception
inception_n
inception_n_top
conv_filter_wise
conv_filter_wise_top


'''

''' 

def relu(x):
    return layers.ReLU()(x)


def hard_sigmoid(x):
    return layers.ReLU(6.)(x + 3.) * (1. / 6.)


def hard_swish(x):
    return layers.Multiply()([layers.Activation(hard_sigmoid)(x), x])


# This function is taken from the original tf repo.
# It ensures that all layers have a channel number that is divisible by 8
# It can be seen here:
# https://github.com/tensorflow/models/blob/master/research/
# slim/nets/mobilenet/mobilenet.py


def _depth(v, divisor=8, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def _se_block(inputs, filters, se_ratio, prefix):
    x = layers.GlobalAveragePooling2D(name=prefix + 'squeeze_excite/AvgPool')(inputs)
    if backend.image_data_format() == 'channels_first':
        x = layers.Reshape((filters, 1, 1))(x)
    else:
        x = layers.Reshape((1, 1, filters))(x)
    x = layers.Conv2D(_depth(filters * se_ratio),
                      kernel_size=1,
                      padding='same',
                      name=prefix + 'squeeze_excite/Conv')(x)
    x = layers.ReLU(name=prefix + 'squeeze_excite/Relu')(x)
    x = layers.Conv2D(filters,
                      kernel_size=1,
                      padding='same',
                      name=prefix + 'squeeze_excite/Conv_1')(x)
    x = layers.Activation(hard_sigmoid)(x)
    if backend.backend() == 'theano':
        # For the Theano backend, we have to explicitly make
        # the excitation weights broadcastable.
        x = layers.Lambda(
            lambda br: backend.pattern_broadcast(br, [True, True, True, False]),
            output_shape=lambda input_shape: input_shape,
            name=prefix + 'squeeze_excite/broadcast')(x)
    x = layers.Multiply(name=prefix + 'squeeze_excite/Mul')([inputs, x])
    return x


def _inverted_res_block(x, expansion, filters, kernel_size, stride,
                        se_ratio, activation, block_id):
                        
    channel_axis = 1 if backend.image_data_format() == 'channels_first' else -1
    shortcut = x
    prefix = 'expanded_conv/'
    infilters = backend.int_shape(x)[channel_axis]
    if block_id:
        # Expand
        prefix = 'expanded_conv_{}/'.format(block_id)
        x = layers.Conv2D(_depth(infilters * expansion),
                          kernel_size=1,
                          padding='same',
                          use_bias=False,
                          name=prefix + 'expand')(x)
        x = layers.BatchNormalization(axis=channel_axis,
                                      epsilon=1e-3,
                                      momentum=0.999,
                                      name=prefix + 'expand/BatchNorm')(x)
        x = layers.Activation(activation)(x)

    if stride == 2:
        x = layers.ZeroPadding2D(padding=correct_pad(backend, x, kernel_size),
                                 name=prefix + 'depthwise/pad')(x)
    x = layers.DepthwiseConv2D(kernel_size,
                               strides=stride,
                               padding='same' if stride == 1 else 'valid',
                               use_bias=False,
                               name=prefix + 'depthwise')(x)
    x = layers.BatchNormalization(axis=channel_axis,
                                  epsilon=1e-3,
                                  momentum=0.999,
                                  name=prefix + 'depthwise/BatchNorm')(x)
    x = layers.Activation(activation)(x)

    if se_ratio:
        x = _se_block(x, _depth(infilters * expansion), se_ratio, prefix)

    x = layers.Conv2D(filters,
                      kernel_size=1,
                      padding='same',
                      use_bias=False,
                      name=prefix + 'project')(x)
    x = layers.BatchNormalization(axis=channel_axis,
                                  epsilon=1e-3,
                                  momentum=0.999,
                                  name=prefix + 'project/BatchNorm')(x)

    if stride == 1 and infilters == filters:
        x = layers.Add(name=prefix + 'Add')([shortcut, x])
    return x

'''