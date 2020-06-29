
# MoveIt
First of all, you have to initialize an instance of **MoveIt** with few parameters:
* **moves_list :** list.
	* a list of moves or directions to pass into the initialization. 
		* Example of moves list: ["up", "right", "down", "left"].
* **img_per_direction (optional) :** int (default: 100)
	* the number of images per move.
	*  this number uncludes both train and test images.
* **count_test_img (optional) :** int (default: 20)
	* the number of test images per move.
	* must be inferior to *img_per_direction*.
	* this number is uncluded among *img_per_direction* number.
* **working_directory (optional) :** str (default: "working/")
	* create a working folder to save the data.
* **train_directory (optional) :** str (default: "train/")
	* create a folder to store the train images, this folder is created inside the the *working_directory*.
* **test_directory (optional) :** str (default: "test/")
	* create a folder to store the test images, this folder is created inside the the *working_directory*.
* **models_directory (optional) :** str (default: "models/")
	* create a folder to save the gererated models, this folder is created inside the the *working_directory*.

## Example of using the library :
* fe
	


# Fucntions

## set_moves()
	function to generate images for each move(direction).
### steps:
	- Turn on the webcam
	- Press the "enter" key to start recording images (you can move while recording to generate multiple positions).
	- Once the record is over, you will see that it has finished, so press again "enter" key to continue the second move.
	- Same insctructions until you finish your moves.
	- The webcam's window will disapear once you finish all directions or if you press "escape" to quit.


## split_data(img_size)
	Split the images into train and test sets according to img_per_direction and count_test_img.
	return x_train, y_train, x_test, y_test.
#### Parameters :
* **img_size :** 	tuple( *height*, *width* )
	* The image's size in which it will be resized.


## train_model( X, Y, validation_data: tuple = None, dropout = 0.0, early_stopping: = True, plot_summary = False, plot_history = False, data_augmentation = False, batch_size = 20, epochs = 20, plot_confusion_matrix = False)
	To train a model on the generated data from the function split_data().
	You can skip this step and train a model on your own.
	return model.
#### Parameters :
* **X :** 	Numpy array or Tensorflow tensor.
	* Input data.
	* You can simply pass the *x_train* generated from the *split_data()* function.
* **Y :** Numpy array or list.
	* Target data.
	* You can simply pass the *y_train* generated from the *split_data()* function.
* **validation_data (optional) :** tuple
	* validation data to pass into the model training.
	* 	You can simply pass the *(x_test, y_test)*, generated from the *split_data()* function.
* **dropout (optional) :** float
	* To activate or not the dropout (to prevent overfitting).
* **early_stopping (optional) :** boolean (default: True)
	* Stops model training when the loss validation is stagnated.
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
	load trained model. you can specify a path to model or use newest_model_in_directory method.
	If both parametrs are speciefied (at the same time), an error occurs
#### Parameters :
* **path_to_model :** 	str
	* path to model to load.
* **newest_model_in_directory :** 	tuple
	* If true, select the latest model in the models directory (the models directory is speciefied in the init method). 
* **validation_data (optional) :** 	tuple.
	* To evaluate model on on data, print loss and accuracy.
	* You can simply pass the *(x_test, y_test)* generated from the *split_data()* function.


## test_record( model, speed, accuracy)
	 Test in real time the model with predictions through the webcam moves. the results are displayed on the screen.

#### Parameters :
* **model :** 	trained model ( "model.h5" for example)
	* Trained model.
* **speed :** 	integer.
	* loop's speed. the lower the value, the faster it is
* ** accuracy :**  float.
	* The minimum accuracy to execute the action
	* You can simply pass the *(x_test, y_test)* generated from the *split_data()* function.

## cam_init()
	 Initialize a webcam instance.
	 return a webcam instance.

## record( webcam_instance, model, speed, accuracy, display_text=True)
	 Test in real time the model with predictions through the webcam moves. the results are displayed on the screen.

#### Parameters :
* **webcam_instance :** 
	* webcam's instance.
* **model :** 	trained model ( "model.h5" for example)
	* Trained model.
* **speed :** 	integer
	* loop's speed. the lower the value, the faster it is.
* **accuracy :**  float.
	* The minimum accuracy to execute the action
	* You can simply pass the *(x_test, y_test)* generated from the *split_data()* function.
* **display_text (optional) :** boolean
	* If True, a text will be displayed at the top left of the screen

## cam_destroy(webcam_instance)
	 Destroy the webcam instance.

#### Parameters :
* **webcam_instance :**
	* Webcam instance
