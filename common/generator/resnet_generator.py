import keras
backend = keras.backend
layers  = keras.layers 
models  = keras.models 
utils   = keras.utils
from keras.layers import Input


def residual_block(x, filters, kernel_size=3, stride=1,
           conv_shortcut=True, name=None):
    """A residual block.

    # Arguments
        x: input tensor.
        filters: integer, filters of the bottleneck layer.
        kernel_size: default 3, kernel size of the bottleneck layer.
        stride: default 1, stride of the first layer.
        conv_shortcut: default True, use convolution shortcut if True,
            otherwise identity shortcut.
        name: string, block label.

    # Returns
        Output tensor for the residual block.
    """
    bn_axis = 3 if backend.image_data_format() == 'channels_last' else 1

    if conv_shortcut is True:
        shortcut = layers.Conv2D(4 * filters, 1, strides=stride,
                                 name=name + '_0_conv')(x)
        shortcut = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                                             name=name + '_0_bn')(shortcut)
    else:
        shortcut = x

    x = layers.Conv2D(filters, 1, strides=stride, name=name + '_1_conv')(x)
    x = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                                  name=name + '_1_bn')(x)
    x = layers.Activation('relu', name=name + '_1_relu')(x)

    x = layers.Conv2D(filters, kernel_size, padding='SAME',
                      name=name + '_2_conv')(x)
    x = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                                  name=name + '_2_bn')(x)
    x = layers.Activation('relu', name=name + '_2_relu')(x)

    x = layers.Conv2D(4 * filters, 1, name=name + '_3_conv')(x)
    x = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                                  name=name + '_3_bn')(x)

    x = layers.Add(name=name + '_add')([shortcut, x])
    x = layers.Activation('relu', name=name + '_out')(x)
    return x





def resnet_n_layers(deep,
                    pyramidal,
                    kernel_size,
                    stride,
                    input_tensor,
                    include_top = True,
                    classes = 2):
    
    input = input_tensor

    first_layer = True
    k_layer = 0 
    #for filters,kernel_size in list(zip(filter_list,kernel_size_list)):
    
    for i in range(deep):
        
        if pyramidal == "increasing":
            filters = min(128,8*(i+1)) 
        if pyramidal == "decreasing":
            filters = min(128,3*(deep-i))
        if pyramidal == "retangular":
            filters = min(128,int(8*deep/2))

        if first_layer:
            first_layer = False

            out = residual_block(input, filters, kernel_size=kernel_size, stride=stride,
                    conv_shortcut=True, name='res1')
        
        else:
            out = residual_block(out, filters, kernel_size=kernel_size, stride=stride,
                    conv_shortcut=True, name=str(filters)+'_'+str(kernel_size)+'_L'+str(k_layer))
            k_layer+=1

    if include_top:
        out = layers.GlobalAveragePooling2D(name='avg_pool')(out)
        out = layers.Dense(classes, activation='softmax', name='probs')(out)

    model = keras.Model(inputs=input, outputs=out, name="res1")
    return model


def resnet_1_layer_filter_wise(filters,kernel_size,input_tensor,stride = 1,include_top = True,classes = 2):

    x = input_tensor

    out = residual_block(x, filters, kernel_size=kernel_size, stride=stride,
            conv_shortcut=True, name='res1')

    if include_top:
        out = layers.GlobalAveragePooling2D(name='avg_pool')(out)
        out = layers.Dense(classes, activation='softmax', name='probs')(out)


    model = keras.Model(inputs=x, outputs=out, name="res1")
    return model


class ResWidhtGenerator:
    def __init__(self,constructor,params):
        self.constructor = constructor
        self.p           = params   

    def Constructor(self,weights=None,input_tensor=(128,128,3),classes = 2):

        return self.constructor(self.p[0],
                                self.p[1],
                                input_tensor,
                                stride = self.p[3],
                                include_top = self.p[2],
                                classes = 2)


class ResDephtGenerator:
    def __init__(self,constructor,params):
        self.constructor = constructor
        self.p           = params   

    def Constructor(self,weights=None,input_tensor=(128,128,3),classes = 2):

        return self.constructor(self.p[0],
                                self.p[1],
                                self.p[2],
                                self.p[3],
                                input_tensor,
                                include_top = self.p[4],
                                classes = 2)
