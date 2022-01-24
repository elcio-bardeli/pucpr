from data        import Data
from grid_search import GridSearch
from models      import Constructor

data        = Data()
constructor = Constructor()
gridSearch  = GridSearch()

all_params  = gridSearch.generate()
name        = gridSearch.names()


for p in all_params:
    
    X_train, Y_train,unique_list_train,X_valid, Y_valid,unique_list_valid,test_data, test_id = data.generate(p)
    model = constructor.build(p,name)

    model.fit(X_train, Y_train, batch_size=32, epochs=2,
                verbose=1, validation_data=(X_valid, Y_valid))

    score = model.evaluate(X_valid, Y_valid, verbose=0)

    print('Score Test log_loss: ', score[0])
    print('Valid accuracy: ', score[1])
    print(p)
    print("______________________________________")


    #result = {key: p[name[key]] for key in name}
    #result['test_log_loss']       = score[0]
    #result['validation_accuracy'] = score[1]

    #print(result)





