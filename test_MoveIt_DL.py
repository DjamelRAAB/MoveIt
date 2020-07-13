from moveIt import MoveIt_ML, MoveIt_DL


if __name__== "__main__":
    
    directions = ["up", "right", "down", "left"]
    move = MoveIt_DL(directions, img_per_direction = 30,  test_size = 0.15)
    move.set_moves()
    x_train, y_train, x_test, y_test = move.split_data()
    model = move.train_model(x_train, y_train, validation_data=(x_test, y_test), dropout=0.15, 
        early_stopping=False, data_augmentation = False, epochs = 2, plot_confusion_matrix= True)
  
    model = move.load_model(newest_model_in_directory=True)
    #model = move.load_model(path_to_model='working/models/model_2020-06-20 00:57_7-2868e-04.h5')
    move.test_record(model, 15, 0.7)
