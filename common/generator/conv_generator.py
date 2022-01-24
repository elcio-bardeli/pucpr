import keras
from keras.layers import Input


def conv_filter_wise(filters, 
                    kernel_size,
                    strides,
                    input_tensor,
                    include_top = True,
                    padding = 'valid',
                    classes = 2):

    input = input_tensor
    out = keras.layers.Conv2D(filters, 
                        kernel_size, 
                        padding=padding,
                        strides = strides,
                        name='single_layer')(input)
    if include_top:
        out = keras.layers.GlobalAveragePooling2D(name='avg_pool')(out)
        out = keras.layers.Dense(classes, activation='softmax', name='probs')(out)

    model = keras.Model(inputs=input, outputs=out, name="conv_single_layer")

    return model


def conv_depth_wise(input_tensor,
                    deep,
                    kernel_size,
                    strides,
                    include_top,
                    padding,
                    pyramidal,
                    classes = 2):

                    
                    
                    

    input       = input_tensor
    first_layer = True
    k_layer     = 0

    for i in range(deep):
        
        if pyramidal == "increasing":
            filters = min(512,32*(i+1)) 
        if pyramidal == "decreasing":
            filters = min(512,32*(deep-i))
        if pyramidal == "retangular":
            filters = min(256,int(32*deep/2))
            
        if first_layer:
            first_layer = False
            out = keras.layers.Conv2D(filters, 
                                kernel_size, 
                                padding=padding,
                                strides = strides,
                                name='conv_layer_'+str(k_layer))(input)
            k_layer+=1
        else:
            out = keras.layers.Conv2D(filters, 
                                kernel_size, 
                                padding=padding,
                                strides = strides,
                                name='conv_layer_'+str(k_layer))(out)
            k_layer+=1


    if include_top:
        out = keras.layers.GlobalAveragePooling2D(name='avg_pool')(out)
        out = keras.layers.Dense(classes, activation='softmax', name='probs')(out)

    model = keras.Model(inputs=input, outputs=out, name="conv_single_layer")

    return model


class WidhtGenerator:
    def __init__(self,constructor,parameters):
        self.constructor = constructor
        self.p  = parameters

    def Constructor(self,weights=None,input_tensor=(128,128,3),classes = 2):
        
        return self.constructor(self.p[0],self.p[1],self.p[2],input_tensor,include_top = self.p[3],padding = self.p[4],classes = 2)


class DeepGenerator:
    def __init__(self,constructor,parameters):
        self.constructor = constructor
        self.p  = parameters

    def Constructor(self,weights=None,input_tensor=(128,128,3),classes = 2):
        
        input_tensor,

        ''' 
        self.p[0] deep,
        self.p[1] kernel_size,
        self.p[2] strides,
        self.p[3] include_top,
        self.p[4] padding,
        self.p[5] pyramidal,
        '''

        return self.constructor(input_tensor,
                                self.p[0],
                                self.p[1],
                                self.p[2],
                                self.p[3],
                                self.p[4],
                                self.p[5],
                                classes = 2)



