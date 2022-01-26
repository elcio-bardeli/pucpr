from modules.data        import Data
from modules.grid_search import GridSearch
from modules.models      import Constructor
from modules.ml_flow     import MlFlow

mlFlow      = MlFlow()
data        = Data()
constructor = Constructor()
gridSearch  = GridSearch()

all_params  = gridSearch.generate()
name        = gridSearch.names()

for i,p in enumerate(all_params):

    print("Completed: ", round(i/len(all_params),2),"% ",i,"out of: ",len(all_params))   
    print("Parameters:", p)

    X_train, Y_train,unique_list_train,X_valid, Y_valid,unique_list_valid,test_data, test_id = data.generate(p)
    model   = constructor.build(p,name)

    history = model.fit(X_train, 
                        Y_train, 
                        batch_size = p[3], 
                        epochs     = (10 if p[5] else 30),
                        verbose    = 1, 
                        validation_data=(X_valid, Y_valid))

    score   = model.evaluate(X_valid, Y_valid, verbose=0)

    mlFlow.log_result(history,score,p,i)



        


