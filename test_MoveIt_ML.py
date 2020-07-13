from moveIt import MoveIt_ML, MoveIt_DL


if __name__== "__main__":
    
    directions = ["up", "right", "down", "left"]
    move = MoveIt_ML(directions, img_per_direction = 30,  test_size = 0.15)
    move.set_moves()
    x_train, y_train, x_test, y_test = move.split_data()
    model = move.train_model(x_train, y_train, validation_data=(x_test, y_test), neighbors = list(range(4, 7)) )
    
    
    model = move.load_model(newest_model_in_directory=True)
    #model = move.load_model(path_to_model='working/models/knn_3n_classifier_2020-06-30 18:54.sav')
    move.test_record(model, 15, 0.5)
