import sys
sys.path.append('../')
from common.generator.generator import Generator
from common.config           import Config
config            = Config()


if config.platform == "raspberry":
    from common.resnet.resnet                             import ResNet50
    from common.resnet.resnet                             import ResNet101
    from common.resnet.resnet                             import ResNet152
    from common.resnet.resnet                             import ResNet50V2
    from common.resnet.resnet                             import ResNet101V2
    from common.resnet.resnet                             import ResNet152V2
    from common.resnet.resnet                             import ResNeXt50
    from common.resnet.resnet                             import ResNeXt101
    from common.efficientnet.efficientnet                 import EfficientNetB0
    from common.efficientnet.efficientnet                 import EfficientNetB1
    from common.efficientnet.efficientnet                 import EfficientNetB2
    from common.efficientnet.efficientnet                 import EfficientNetB3
    from common.efficientnet.efficientnet                 import EfficientNetB4
    from common.efficientnet.efficientnet                 import EfficientNetB5
    from common.efficientnet.efficientnet                 import EfficientNetB6
    from common.efficientnet.efficientnet                 import EfficientNetB7
    from common.densenet.densenet                         import DenseNet121
    from common.densenet.densenet                         import DenseNet169
    from common.densenet.densenet                         import DenseNet201
    from common.xception.xception                         import Xception
    from common.inception_resnet_v2.inception_resnet_v2   import InceptionResNetV2
    from common.inception_v3.inception_v3                 import InceptionV3
    from common.mobilenet_v2.mobilenet_v2                 import MobileNetV2
    from common.mobilenet_v3.mobilenet_v3                 import MobileNetV3Small
    from common.mobilenet_v3.mobilenet_v3                 import MobileNetV3Large
    from common.mobilenet.mobilenet                       import MobileNet 
    from common.nasnet.nasnet                             import NASNet
    from common.nasnet.nasnet                             import NASNetLarge
    from common.nasnet.nasnet                             import NASNetMobile
    from common.vgg16.vgg16                               import VGG16
    from common.vgg19.vgg19                               import VGG19

    constructor = {
        "VGG16":              VGG16,                     
        "VGG19":              VGG19, 
        "ResNet50":           ResNet50,                        
        "ResNet101":          ResNet101,                         
        "ResNet152":          ResNet152,                         
        "ResNet50V2":         ResNet50V2,                          
        "ResNet101V2":        ResNet101V2,                           
        "ResNet152V2":        ResNet152V2,                           
        "ResNeXt50":          ResNeXt50,                         
        "ResNeXt101":         ResNeXt101,
        "EfficientNetB0":     EfficientNetB0,                              
        "EfficientNetB1":     EfficientNetB1,                              
        "EfficientNetB2":     EfficientNetB2,                              
        "EfficientNetB3":     EfficientNetB3,                              
        "EfficientNetB4":     EfficientNetB4,                              
        "EfficientNetB5":     EfficientNetB5,                              
        "EfficientNetB6":     EfficientNetB6,                              
        "EfficientNetB7":     EfficientNetB7,                              
        "DenseNet121":        DenseNet121,                           
        "DenseNet169":        DenseNet169,                           
        "DenseNet201":        DenseNet201,                           
        "Xception":           Xception,                        
        "InceptionResNetV2":  InceptionResNetV2,                                 
        "InceptionV3":        InceptionV3,                           
        "MobileNetV2":        MobileNetV2,                           
        "MobileNetV3Small":   MobileNetV3Small,                                
        "MobileNetV3Large":   MobileNetV3Large,                                
        "MobileNet":          MobileNet,                         
        "NASNet":             NASNet,                      
        "NASNetLarge":        NASNetLarge,                           
        "NASNetMobile":       NASNetMobile                                                                
    }


else:

    from tensorflow.keras.applications.efficientnet  import EfficientNetB0
    from tensorflow.keras.applications.efficientnet  import EfficientNetB1
    from tensorflow.keras.applications.efficientnet  import EfficientNetB2     
    from tensorflow.keras.applications.efficientnet  import EfficientNetB3
    from tensorflow.keras.applications.efficientnet  import EfficientNetB4
    from tensorflow.keras.applications.efficientnet  import EfficientNetB5
    from tensorflow.keras.applications.efficientnet  import EfficientNetB6
    from tensorflow.keras.applications.efficientnet  import EfficientNetB7
    from tensorflow.keras.applications.mobilenet_v2  import MobileNetV2
    from tensorflow.keras.applications.mobilenet     import MobileNet
    from tensorflow.keras.applications.densenet      import DenseNet121
    from tensorflow.keras.applications.densenet      import DenseNet169
    from tensorflow.keras.applications.densenet      import DenseNet201
    from tensorflow.keras.applications.nasnet        import NASNetMobile
    from tensorflow.keras.applications.nasnet        import NASNetLarge
    from tensorflow.keras.applications.vgg16         import VGG16
    from tensorflow.keras.applications.vgg19         import VGG19
    from tensorflow.keras.applications.xception      import Xception
    from tensorflow.keras.applications.resnet50      import ResNet50
    from tensorflow.keras.applications               import ResNet101
    from tensorflow.keras.applications               import ResNet152
    from tensorflow.keras.applications               import ResNet50V2
    from tensorflow.keras.applications               import ResNet101V2
    from tensorflow.keras.applications               import ResNet152V2
    from tensorflow.keras.applications               import InceptionV3
    from tensorflow.keras.applications               import InceptionResNetV2


    constructor = {
        "EfficientNetB0":      EfficientNetB0,              
        "EfficientNetB1":      EfficientNetB1,              
        "EfficientNetB2":      EfficientNetB2,              
        "EfficientNetB3":      EfficientNetB3,              
        "EfficientNetB4":      EfficientNetB4,              
        "EfficientNetB5":      EfficientNetB5,              
        "EfficientNetB6":      EfficientNetB6,              
        "EfficientNetB7":      EfficientNetB7,              
        "MobileNetV2":         MobileNetV2,           
        "MobileNet":           MobileNet,         
        "DenseNet121":         DenseNet121,           
        "DenseNet169":         DenseNet169,           
        "DenseNet201":         DenseNet201,           
        "NASNetMobile":        NASNetMobile,            
        "NASNetLarge":         NASNetLarge,           
        "VGG16":               VGG16,     
        "VGG19":               VGG19,     
        "Xception":            Xception,        
        "ResNet50":            ResNet50,        
        "ResNet101":           ResNet101,         
        "ResNet152":           ResNet152,         
        "ResNet50V2":          ResNet50V2,          
        "ResNet101V2":         ResNet101V2,           
        "ResNet152V2":         ResNet152V2,           
        "InceptionV3":         InceptionV3,           
        "InceptionResNetV2":   InceptionResNetV2                                                                                
    }



generator    = Generator()
constructors = generator.run()

constructor.update(constructors)
#print(len(constructor))