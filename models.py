import tensorflow as tf



class Constructor:
    def __init__(self) -> None:
        
        self.constructors = {"Vgg16":       tf.keras.applications.vgg16.VGG16,
                             "ResNet50":    tf.keras.applications.ResNet50,
                             "InceptionV3": tf.keras.applications.InceptionV3}

    def build(self,params,name):
        
        backbone = self.constructors[params[name["backbone"]]]
        backbone = backbone(include_top= False, 
                    weights    = None,
                    input_shape= (params[0][1],params[0][0],3))

        #backbone.summary()

        x = backbone.output
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dropout(0.5)(x)

        output = tf.keras.layers.Dense(units=10, activation=tf.nn.softmax)(x)
        model = tf.keras.models.Model(inputs=backbone.inputs, outputs=output)

        model.compile(optimizer=tf.keras.optimizers.Adam(params[4]), #0.0001
                        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
                            metrics=['accuracy'])

        return model