import mlflow


class MlFlow:
    def __init__(self) -> None:
        mlflow.set_experiment(experiment_name ='experiment_video')
        
    def log_result(self,history,score,paramters,i):

        ''' 
        "img_size":          0,
        "img_normalization": 1,
        "backbone":          2,
        "batch_size":        3,
        "learning_rate":     4,
        "pre_treined":       5
        '''

        with mlflow.start_run(run_name = str(i)):
            
            mlflow.log_params({"img_size":          paramters[0],
                               "img_normalization": paramters[1],
                               "backbone":          paramters[2],
                               "batch_size":        paramters[3],
                               "learning_rate":     paramters[4],
                               "pre_treined":       paramters[5]})

            loss     = history.history['loss']       
            acc      = history.history['acc']        
            val_loss = history.history['val_loss']   
            val_acc  = history.history['val_acc']    

            mlflow.log_metric(key='Test_log_loss', value = score[0])
            mlflow.log_metric(key='Valid accuracy', value = score[1])

            
            for i in range(len(loss)):
                
                mlflow.log_metric(key='loss',     value = loss[i],     step=i) 
                mlflow.log_metric(key='acc',      value = acc[i],      step=i)
                mlflow.log_metric(key='val_loss', value = val_loss[i], step=i)
                mlflow.log_metric(key='val_acc',  value = val_acc[i],  step=i)
