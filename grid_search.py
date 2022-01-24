import itertools

class GridSearch:
    def __init__(self) -> None:
        
        self.img_size          = [(96, 128),(128, 160),(160, 224)]
        self.img_normalization = ['0_1','standardzation','none']
        self.models            = ["Vgg16","ResNet50","InceptionV3"]
        
        batch_size        = [1,2,3,5,8,10]
        learning_rate     = [0.01,0.001,0.0001]
        pre_treined       = [None,'imagenet']

    def generate(self) -> list:
        all_params = list(itertools.product(self.img_size,
                                            self.img_normalization,
                                            self.models))
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
    

