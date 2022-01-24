import keras
from keras.layers import Input

def conv2d_bn(x,
              filters,
              num_row,
              num_col,
              padding='same',
              strides=(1, 1),
              name=None):
    """Utility function to apply conv + BN.

    # Arguments
        x: input tensor.
        filters: filters in `Conv2D`.
        num_row: height of the convolution kernel.
        num_col: width of the convolution kernel.
        padding: padding mode in `Conv2D`.
        strides: strides in `Conv2D`.
        name: name of the ops; will become `name + '_conv'`
            for the convolution and `name + '_bn'` for the
            batch norm layer.

    # Returns
        Output tensor after applying `Conv2D` and `BatchNormalization`.
    """
    if name is not None:
        bn_name = name + '_bn'
        conv_name = name + '_conv'
    else:
        bn_name = None
        conv_name = None
    if keras.backend.image_data_format() == 'channels_first':
        bn_axis = 1
    else:
        bn_axis = 3
    x = keras.layers.Conv2D(
        filters, (num_row, num_col),
        strides=strides,
        padding=padding,
        use_bias=False,
        name=conv_name)(x)
    x = keras.layers.BatchNormalization(axis=bn_axis, scale=False, name=bn_name)(x)
    x = keras.layers.Activation('relu', name=name)(x)
    return x

def inception_1_layer(input_shape,include_top = True,classes=2):
    img_input = Input(shape=input_shape)

    if keras.backend.image_data_format() == 'channels_first':
        channel_axis = 1
    else:
        channel_axis = 3

    x = conv2d_bn(img_input, 32, 3, 3, strides=(2, 2), padding='valid')
    x = conv2d_bn(x, 32, 3, 3, padding='valid')
    x = conv2d_bn(x, 64, 3, 3)
    x = keras.layers.MaxPooling2D((3, 3), strides=(2, 2))(x)
    x = conv2d_bn(x, 82, 1, 1, padding='valid')
    x = conv2d_bn(x, 192, 3, 3, padding='valid')
    x = keras.layers.MaxPooling2D((3, 3), strides=(2, 2))(x)

    # mixed 0: 35 x 35 x 256
    branch1x1 = conv2d_bn(x, 64, 1, 1)

    branch5x5 = conv2d_bn(x, 48, 1, 1)
    branch5x5 = conv2d_bn(branch5x5, 64, 5, 5)

    branch3x3dbl = conv2d_bn(x, 64, 1, 1)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)

    branch_pool = keras.layers.AveragePooling2D((3, 3),
                                          strides=(1, 1),
                                          padding='same')(x)
    branch_pool = conv2d_bn(branch_pool, 32, 1, 1)
    
    x = keras.layers.concatenate(
        [branch1x1, branch5x5, branch3x3dbl, branch_pool],
        axis=channel_axis,
        name='mixed0')

    
    #if include_top:
    #    x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
    #    x = layers.Dense(classes, activation='softmax', name='probs')(x)

    model = keras.Model(inputs=img_input, outputs=x, name="inception")

    return model


def inception_n_layers(shape_1x1,
                        shape_5x5,
                        shape_3x3dbl,
                        shape_pool,
                        input_tensor,
                        include_top = True,
                        classes=2):

    img_input = input_tensor

    if keras.backend.image_data_format() == 'channels_first':
        channel_axis = 1
    else:
        channel_axis = 3

    
    first_layer = True
    k_layer = 0
    for _1x1,_5x5,_3x3dbl,_pool in list(zip(shape_1x1,shape_5x5,shape_3x3dbl,shape_pool)):
        if first_layer:
            first_layer = False
            x = conv2d_bn(img_input, 32, 3, 3, strides=(2, 2), padding='valid') #31
            
            x = conv2d_bn(x, 32, 3, 3, padding='valid') #29
            x = conv2d_bn(x, 64, 3, 3)# 29
            x = keras.layers.MaxPooling2D((3, 3), strides=(2, 2))(x) # 14
            x = conv2d_bn(x, 82, 1, 1, padding='valid') #14
            x = conv2d_bn(x, 192, 3, 3, padding='valid') #12
            x = keras.layers.MaxPooling2D((3, 3), strides=(2, 2))(x)
            
            # mixed 0: 35 x 35 x 256
            branch1x1 = conv2d_bn(x, _1x1, 1, 1)

            branch5x5 = conv2d_bn(x, 48, 1, 1)
            branch5x5 = conv2d_bn(branch5x5, _5x5, 5, 5)

            branch3x3dbl = conv2d_bn(x, 64, 1, 1)
            branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)
            branch3x3dbl = conv2d_bn(branch3x3dbl, _3x3dbl, 3, 3)

            branch_pool = keras.layers.AveragePooling2D((3, 3),
                                                strides=(1, 1),
                                                padding='same')(x)
            branch_pool = conv2d_bn(branch_pool, _pool, 1, 1)
            
            x = keras.layers.concatenate(
                [branch1x1, branch5x5, branch3x3dbl, branch_pool],
                axis=channel_axis,
                name='mixed'+str(k_layer))
            k_layer+=1

             

        else:
            branch1x1 = conv2d_bn(x, _1x1, 1, 1)
            branch5x5 = conv2d_bn(x, 48, 1, 1)
            branch5x5 = conv2d_bn(branch5x5, _5x5, 5, 5)

            branch3x3dbl = conv2d_bn(x, 64, 1, 1)
            branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)
            branch3x3dbl = conv2d_bn(branch3x3dbl, _3x3dbl, 3, 3)

            branch_pool = keras.layers.AveragePooling2D((3, 3),
                                                strides=(1, 1),
                                                padding='same')(x)
            branch_pool = conv2d_bn(branch_pool, _pool, 1, 1)
            x = keras.layers.concatenate(
                [branch1x1, branch5x5, branch3x3dbl, branch_pool],
                axis=channel_axis,
                name='mixed'+str(k_layer))
            k_layer+=1

    
    if include_top:
        x = keras.layers.GlobalAveragePooling2D(name='avg_pool')(x)
        x = keras.layers.Dense(classes, activation='softmax', name='probs')(x)

    model = keras.Model(inputs=img_input, outputs=x, name="inception")

    return model



class InceptionGenerator:
    def __init__(self,constructor,
                    shape_1x1,
                    shape_5x5,
                    shape_3x3dbl,
                    shape_pool,
                    include_top):

        self.constructor  = constructor
        self.shape_1x1    = shape_1x1               
        self.shape_5x5    = shape_5x5               
        self.shape_3x3dbl = shape_3x3dbl                  
        self.shape_pool   = shape_pool 
        self.include_top  = include_top             

    def Constructor(self,weights=None,input_tensor=(128,128,3),classes = 2):
        
        return self.constructor(self.shape_1x1,
                                self.shape_5x5,
                                self.shape_3x3dbl,
                                self.shape_pool,
                                input_tensor,
                                include_top = self.include_top,
                                classes = 2)