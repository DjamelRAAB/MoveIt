

# MoveIt
First of all, you have to initialize an instance of **MoveIt_ML** or **MoveIt_DL** with few parameters:
* **moves_list :** list.
	* a list of moves or directions to pass during initialization. 
		* Example of moves list: ["up", "right", "down", "left"].
* **img_per_direction (optional) :** int (default: 200)
	* number of images per move.
	*  this number includes both train and test images.
	*  the larger the number, the better
*  **mask_x, mask_y (optional) :** int (default: 200, 100 respectively)
	* the top left coordinates of the rectangle displayed on the webcam.
*  **mask_height,  mask_width (optional) :** int (default: 250, 220 respectively)
	* the top left coordinates of the rectangle displayed on the webcam.
* **test_size (optional) :** float (default: 0.2)
	* percent of images intended for evaluation of the model.
* **working_directory (optional) :** str (default: "working/")
	* workspace folder's name.
* **train_directory (optional) :** str (default: "train/")
	* train image's folder, it's created inside the *working_directory*.
* **test_directory (optional) :** str (default: "test/")
	* test image's folder, it's created inside the *working_directory*.
* **models_directory (optional) :** str (default: "models/")
	* trained model's folder, it's created inside the *working_directory*.

### Here an example of each one of them :
	from moveit import MoveIt_ML, MoveIt_DL
	
	directions = ["up", "right", "down", "left"]
	moveit = MoveIt_ML(directions, img_per_direction = 300, test_size = 0.15)
	# or
	# moveit = MoveIt_DL(directions, img_per_direction = 300, test_size = 0.15)

# Functions

## set_moves()
	Takes shots from the webcam and save them in folders.
	 each direction is recorded in a folder bearing its name.
### steps:
	- Turn on the webcam
	- Press the "enter" key to start recording images (you should move while recording to generate differents shot's move).
	- Once the record is over, you will be notified, then press again "enter" key to continue the second move.
	- Same insctructions until you finish your moves.
	- The webcam's window will disapear once you finish all directions record.


## split_data(img_size)
	Split the images into train and test sets according to *img_per_direction* and *test_size*.
	return x_train, y_train, x_test, y_test.
 #### Parameters :
* **img_size (optional):** 	tuple(*height*, *width*)
	* Images' size in which it will be resized.

## MoveIt_ML.train_model( X, Y, validation_data, neighbors = 5, best_score = True, plot_history = True, plot_confusion_matrix = True)
	To train a model on the generated data received from the function split_data(), with K-nearest neighbors algorithm.
	You can skip this step and train a model on your own (must be a ML algorithm).
	returns model.
#### Parameters :
* **X :** 	Numpy array.
	* Input data.
	* You can simply pass the *x_train* generated from the *split_data()* function.
* **Y :** Numpy array or list.
	* Target data.
	* You can simply pass the *y_train* generated from the *split_data()* function.
* **validation_data :** tuple
	* validation data to evaluate the model during training.
	* 	You can simply pass the *(x_test, y_test)*, generated from the *split_data()* function.
* ** neighbors (optional) : ** int or list[int] (default: 5)
	* Number of neighbors used to train the model.
	* if list is passed, it will be evaluated on each of them and the best validation accuracy is returned (or the first one if similar).
* **best_score (optional) :** boolean (default: True)
	* if true, retrain the model with the best score's neighbor.
* **plot_history (optional) :** boolean (default: False)
	* plot history of both accuracy and loss train (and validation if filled).
* **plot_confusion_matrix (optional) :** boolean (default: False)
	* plot model's confusion matrix.

#### Example :
	model = moveit.train_model(x_train, y_train, validation_data=(x_test, y_test), neighbors = list(range(4, 9)) )


## MoveIt_DL.train_model( X, Y, validation_data = None, dropout = 0.0, early_stopping = True, plot_summary = False, plot_history = False, data_augmentation = False, batch_size = 20, epochs = 20, plot_confusion_matrix = False)
	To train a model on the generated data from the function *split_data()*, with a convolutional neural network. Here is the model architecture :
	<img src="model architecture">
	You can skip this step and train a model on your own.
	return model.
#### Parameters :
* **X :** 	Numpy array or TensorFlow tensor.
	* Input data.
	* You can simply pass the *x_train* generated from the *split_data()* function.
* **Y :** Numpy array or list.
	* Target data.
	* You can simply pass the *y_train* generated from the *split_data()* function.
* **validation_data (optional) :** tuple
	* validation data to evaluate the model during training.
	* 	You can simply pass the *(x_test, y_test)*, generated from the *split_data()* function.
* **dropout (optional) :** float
	* To activate or not the dropout (to prevent overfitting).
* **early_stopping (optional) :** boolean (default: True)
	* Stops model training when the loss validation is stagnated or still decreasing after 3 epochs.
* **plot_summary (optional) :** boolean (default: False)
	* Display the model's architecture.
* **plot_history (optional) :** boolean (default: False)
	* plot history of both accuracy and loss train (and validation if filled).
* **plot_confusion_matrix (optional) :** boolean (default: False)
	* plot model's confusion matrix.
* **data_augmentation (optional) :** boolean (default: False)
	* Activate data augmentation when training model (to prevent overfitting).
* **batch_size (optional) :** int (default: 20)
	* model's batch size when fitting.
* **epochs (optional) :** int (default: 20)
	* Number of epochs of fitting.
	* Can be stopped earlier if *early_stopping* is activated.

#### Example :
	model = moveIt.train_model(x_train, y_train, validation_data=(x_test, y_test), dropout=0.1, early_stopping=True, data_augmentation = False, epochs = 20, plot_confusion_matrix= True)



## load_model(path_to_model = None, newest_model_in_directory = False, validation_data = None)
	to load a trained model. you can specify a path to model or use *newest_model_in_directory* method.
	If both parameters are speciefied (at the same time), an error occurs.
#### Parameters :
* **path_to_model :** 	str
	* path to model to load.
	*  MoveIt_ML extension : model_name.sav
	* MoveIt_DL extension : model_name.h5
* **newest_model_in_directory :** 	boolean (default: False)
	* If true, select the latest model in the models' directory (the models' directory is specified during initialization). 
* **validation_data (optional) :** 	tuple.
	* To evaluate the model, prints loss and accuracy.
	* You can simply pass the *(x_test, y_test)* generated from the *split_data()* function.


## test_record( model, speed, threshold)
	 Test and evaluate the loaded model in real time through the webcam moves. the results are displayed on the screen.

#### Parameters :
* **model :** 	trained model ( "model.h5" or "model.sav" for example)
	* Trained model.
* **speed :** 	integer.
	* loop's speed. The lower the value, the faster it is
* **threshold :** float.
	* The minimum accuracy to execute the action.

## show_rectangle(frame)
	 To use when starting webcam instance with open-cv (place it before cv2.imshow()). 
	 It can be used when deploying the model.
	 returns the same frame with rectangle at the middle.
#### Parameters :
* **frame :** 	image.
	* generated image by the webcam.

#### Example :
	moveit = MoveIt_ML(list_of_directions)
	capture = cv2.VideoCapture(0)
	while True:
            ret, frame = capture.read()
            frame = moveit.show_rectangle(frame)
	        cv2.imshow("VIDEO", frame)

## cam_init(frame, model, threshold)
	 
	To init and create a cam instance
	returns cv2.VideoCapture(0) instance


## predict(cam, frame, model, threshold, prev_class=None, prev_rate=None)
	 Test in real time the model with predictions through the webcam moves.
	 You can even check how to use this function via the source code of the applications on github (examples).
	 returns class_name, class_predicted_rate

#### Parameters :
* **cam :** webcam instance (cv2.VideoCapture(0)).
* **model :** 	trained model ("model.h5" or "model.sav" for example).
* **speed :** int.
	* loop's speed.
* **threshold :**  float between [0, 1].
	* The minimum accuracy to execute the action. If the predicted accuracy is lower than the threshold, it returns None.
* **prev_class :** str.
	* The previous prediction class, to show on screen.
* **prev_rate :** float.
	* The previous prediction accuracy, to show on screen.

## cam_destroy(webcam_instance)
	 To destroy and release the cam instance.

#### Parameters :
* **webcam_instance :** an instance created when initializing cv2.VideoCapture(0).


## Example of using MoveIt_ML :
	from moveIt import MoveIt_ML
	
	directions = ["up", "right", "down", "left"]
    move = MoveIt_ML(directions, img_per_direction = 300,  test_size = 0.15)
    move.set_moves()
    x_train, y_train, x_test, y_test = move.split_data()
    model = move.train_model(x_train, y_train, validation_data=(x_test, y_test), neighbors = list(range(3, 8)) )
    
    model = move.load_model(newest_model_in_directory=True)
    # or use model = move.load_model(path_to_model='working/models/model_name.sav')
    move.test_record(model, 15, 0.5)
   
   

## Example of using MoveIt_DL :
	from moveIt import MoveIt_DL
	
	directions = ["up", "right", "down", "left"]
    move = MoveIt_ML(directions, img_per_direction = 300,  test_size = 0.15)
    move.set_moves()
    x_train, y_train, x_test, y_test = move.split_data()
    model = move.train_model(x_train, y_train, validation_data=(x_test, y_test), dropout=0.1, early_stopping=False, data_augmentation = False, epochs = 10, plot_confusion_matrix= True)
    
    model = move.load_model(newest_model_in_directory=True)
    # or use model = move.load_model(path_to_model='working/models/model_name.h5')
    move.test_record(model, 15, 0.5)

## Example of deploying the *Predict ()* function :
	from moveIt import MoveIt_DL
	
	model = moveit.load_model(path_to_model='working/models/model_name.sav')
    cam = moveit.cam_init()
    direction, acc = None, 0
    i = 0
    while True:
        direction, acc = moveit.predict(cam, model, 25, 0.5, prev_class = direction, prev_rate = acc)
        print("direction %s : %s (%.0f%%) " %(i, direction, acc*100) )
        if direction is not None:
            key_down(direction)
        i += 1

    moveit.cam_destroy(cam)

