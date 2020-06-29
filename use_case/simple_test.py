from moveit.ml import *
from matplotlib import pyplot as plt


if __name__== "__main__":
    directions = ["up", "right", "down", "left"]
    move = MoveIt_ML(directions)
    print(move)
    move.set_moves()
    #x_train, y_train, x_test, y_test = move.split_data(img_size= (60, 105))

    #model = move.train_model(x_train, y_train, validation_data=(x_test, y_test), dropout=0.2, 
    #    early_stopping=True, data_augmentation = False, epochs = 50, plot_confusion_matrix= True)
    #model = move.train_model(x_train, y_train, validation_data=(x_test, y_test), dropout=0.2, 
    #    early_stopping=True, data_augmentation = True, epochs = 50, plot_confusion_matrix= True)
    
    #model = move.load_model(newest_model_in_directory=True)
    #plt.imshow(x_train[0])
    #plt.show()
    #model = move.load_model(path_to_model='working/models/model_2020-05-14 00:45.h5', validation_data=(x_test, y_test))
    #model = move.load_model(newest_model_in_directory=True, path_to_model='working/models/model_2020-05-12 00:18.h5', validation_data=(x_test, y_test))
    #move.record(model, 15, 0.95)
