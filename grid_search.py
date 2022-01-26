import itertools

class GridSearch:
    def __init__(self) -> None:
        #480x640
        
        self.img_size          = [(96, 128),(128, 160),(160, 224)] 
        self.img_normalization = ['0_1','none','standardzation'] 
        self.models            = ["InceptionV3","Vgg16","ResNet50"]
        
        self.batch_size        = [4,8,16] 
        self.learning_rate     = [0.0001] #0.01,0.001,
        self.pre_treined       = ['imagenet',None] #,

    def generate(self) -> list:
        all_params = list(itertools.product(self.img_size,
                                            self.img_normalization,
                                            self.models,
                                            self.batch_size, 
                                            self.learning_rate,
                                            self.pre_treined))
        return all_params

    def names(self):

        name = {
            "img_size":          0,
            "img_normalization": 1,
            "backbone":          2,
            "batch_size":        3,
            "learning_rate":     4,
            "pre_treined":       5
        }

        return name


#(include_top=False, weights='imagenet',input_shape=(self.img_rows, self.img_cols, color_type_global))
    

