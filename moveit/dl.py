import cv2
import os
import time
import datetime
import numpy as np
import shutil
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.activations import *
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


"""
nbr_classes=180 # histSize 256
hist_size = 180   
demo = 0
THRESHOLD=1
objet=0
mask=0 
mask_x=180
mask_y=100
mask_width=220
mask_height=250
"""

class MoveIt_DL:
    
    def __init__(self, moves_list: list, img_per_direction: int = 100,
                 test_size: float = 0.2, mask_x=180, mask_y=100, mask_width=220, 
                 mask_height=250, working_directory: str = "working/", 
                 train_directory: str = "train/", test_directory: str = "test/", 
                 models_directory: str = "models/"):

        self.directions = moves_list
        self.img_size = None
        self.img_count = img_per_direction
        self.test_size = test_size
        self.mask_x, self.mask_y, self.mask_width, self.mask_height =  mask_x, mask_y, mask_width, mask_height
        self.hist_size = 180
        self.working_directory = working_directory
        self.train_directory = working_directory + train_directory
        self.test_directory = working_directory + test_directory
        self.models_directory = working_directory + models_directory
        self.mask_path = working_directory + "mask.jpg"

        if not os.path.exists(self.working_directory): os.makedirs(self.working_directory)
        if not os.path.exists(self.train_directory): os.makedirs(self.train_directory)
        if not os.path.exists(self.test_directory): os.makedirs(self.test_directory)
        if not os.path.exists(self.models_directory): os.makedirs(self.models_directory)
        
        print("moveIt ready to use !")

    def __str__(self):
        return 'MoveIt_DL class: \nmoves: %s\n image size: %s\n image per move: %s\n test image per move: %s\n' \
                 %(self.directions, str(self.img_size), self.img_count, self.test_size)

    def __webcam_with_text(self, frame, text_list: list, display_text=True):
        if display_text:

            cv2.putText(frame, "press escape to quit", (0, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 0, 255))
            org_height = 20
            for text in text_list:
                cv2.putText(frame, text, (0, org_height), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (0, 180, 255), thickness=2, lineType=cv2.LINE_AA)
                org_height += org_height
        cv2.imshow('VIDEO', frame)


    def __define_mask(self, frame, directory, resized = False):        
        if not resized:
            frame = frame[self.mask_y: self.mask_y + self.mask_height, self.mask_x: self.mask_x + self.mask_width]
            cv2.imwrite(directory, img=frame)
        # transform the image FROM BGR to HSV(hue saturation value)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # Histogram of an instensity distribution of an image
        ch = (0, 0)
        hue = np.empty(hsv.shape, hsv.dtype)
        cv2.mixChannels([hsv], [hue], ch)
        roi_hist = cv2.calcHist([hue], [0], None, [self.hist_size], [0, self.hist_size], accumulate=False)
        cv2.normalize(roi_hist, roi_hist, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

        return roi_hist

    def __show_mask(self, frame, roi_hist):
        margin = 2
         #frame = frame[mask_y + margin: mask_y + mask_height -margin, mask_x + margin: mask_x + mask_width - margin]
        frame = frame[self.mask_y : self.mask_y + self.mask_height, self.mask_x : self.mask_x + self.mask_width]
        frame=cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # Back projection of a histogram
        mask=cv2.calcBackProject([frame], [0], roi_hist, [0, self.hist_size], scale=1)
        _, mask2 = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)
        # shrink the object, then dilate it
        mask3=cv2.erode(mask2, None, iterations=3)
        cv2.imshow("mask3", mask3)
        mask4=cv2.dilate(mask3, None, iterations=3)
        cv2.imshow("mask4", mask4)
        mask5 = cv2.medianBlur(mask4, 19)
        return mask5

    # plot two separable plots of loss/accuracy model
    def __history_model(self, history):
        plt.subplot(121)
        plt.plot(history['accuracy'])
        plt.plot(history['val_accuracy'])
        plt.title('Model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epochs')
        plt.ylim([0.0, 1.0])
        plt.legend(['train_accuracy', 'val_accuracy'], loc='upper left')

        plt.subplot(122)
        plt.plot(history['loss'])
        plt.plot(history['val_loss'])
        plt.title('Model loss')
        plt.ylabel('loss')
        plt.xlabel('epochs')
        plt.ylim([0.0, 3.0])
        plt.legend(['train_loss', 'val_loss'])

        plt.show()

    def __confusion(self, directions, true_targets, predicted_targets):

        cm = confusion_matrix(true_targets, predicted_targets)
        fig, ax = plt.subplots(1, 1, figsize=(6, 5))
        ax = sns.heatmap(cm, annot=True, fmt="d", cmap='YlGnBu')
        plt.tight_layout()
        tick_marks = np.arange(len(directions))
        plt.xticks(tick_marks, directions)
        plt.yticks(tick_marks, directions)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.show()


    def __newest(self, path: str):

        files = os.listdir(path)
        paths = [os.path.join(path, basename) for basename in files]
        return max(paths, key=os.path.getctime)
    

    def set_moves(self, show_mask = True):
        #global demo
        
        # Initialize the webcam
        cap = cv2.VideoCapture(0)
        time.sleep(0.5)

        directions_counter = 0
        count = 0
        taking_picture = False
        back_proj = False

        while True:
            # starting the webcam
            ret, frame = cap.read()
            
            cv2.rectangle(frame, (self.mask_x, self.mask_y), (self.mask_x+self.mask_width, self.mask_y+self.mask_height), (255,0,0), 2)
            if taking_picture:
                cv2.imshow("VIDEO", frame)
                if back_proj:
                    mask = self.__show_mask(frame, roi_hist)
                    cv2.imshow("mask", mask)

            else:
                if back_proj:
                    mask = self.__show_mask(frame, roi_hist)
                    cv2.imshow("mask", mask)
                    if directions_counter != 0:
                        self.__webcam_with_text(frame, ["Move {} succefully saved !".format(self.directions[directions_counter -1]),
                                                        "Move {}: press enter to start !".format(self.directions[directions_counter ])])
                    else:    
                        self.__webcam_with_text(frame, ["Move {}: press enter to start !".format(self.directions[directions_counter ])])
                else: 
                    self.__webcam_with_text(frame, ["Place your move inside the rectangle then press enter"])   
                    #cv2.rectangle(frame, (mask_x, mask_y), (mask_x+mask_width, mask_y+mask_height), (255,0,0), 2)
                    
            
            key = cv2.waitKey(1)
            if key == 13:  # 13 is the Enter key
                if back_proj:
                    taking_picture = True
                    count = 0
                    direction = self.directions[directions_counter]
                    print("starting pictures taking ({}) ...".format(direction))
                else:
                    roi_hist = self.__define_mask(frame, self.mask_path)
                    back_proj = True

            if taking_picture:
                #  create the folders
                direction_folder = self.working_directory + self.directions[directions_counter]
                if not os.path.exists(direction_folder): os.makedirs(direction_folder)
                if count < self.img_count:
                    filename = direction_folder + "/" + self.directions[directions_counter] + "_" + str(count) + ".jpg"
                    image_to_save = self.__show_mask(frame, roi_hist)
                    cv2.imwrite(filename, img=image_to_save)
                    count += 1

                else:
                    directions_counter += 1
                    taking_picture = False
                    print("{} pictures imported in '{}' folder".format(self.img_count, direction))

            if key == 27:
                print("Stopping webcam ...")
                break

            if directions_counter > len(self.directions) - 1:  # stop when all the self.directions are set
                print("==============================================")
                print("all directions have been recorded successfully")
                print("==============================================")
                break
        
        # Release the webcam
        cap.release()
        cv2.destroyAllWindows()


    def split_data(self, img_size: tuple = None):
        
        if img_size is not None:
            self.img_size = img_size[:2]
        else:
            # get the shape of the first image
            first_dir = self.working_directory + self.directions[0] + "/"
            self.img_size = cv2.imread(first_dir  + os.listdir(first_dir)[0]).shape[:2]

        if len(os.listdir(self.train_directory)) == 0:
            try:
                for direction in range(len(self.directions)):
                    dir_path = self.working_directory + self.directions[direction] + "/"
                    dir_images_names = os.listdir(dir_path)

                    train_images_names, test_images_names = train_test_split(dir_images_names, test_size= self.test_size)
                    # moving to TEST folder ELSE train
                    for train_img in train_images_names:
                        source_file =  dir_path + train_img
                        dest_file =  self.train_directory + train_img
                        shutil.copy2(source_file, dest_file)

                    for test_img in test_images_names:
                        source_file =  dir_path + test_img
                        dest_file =  self.test_directory + test_img
                        shutil.copy2(source_file, dest_file)
                print("the files have been moved successfully !")
            except:
                print("Warning: error occured during moved the files !")
        else:
            print("the files have already been moved !")

        TRAIN_IMG = os.listdir(self.train_directory)
        TEST_IMG = os.listdir(self.test_directory)
        directions_dict = dict(zip(self.directions, range(len(self.directions))))

        x_train = np.ndarray(shape=(len(TRAIN_IMG), self.img_size[0], self.img_size[1]),
                             dtype=float)
        y_train = np.ndarray(shape=(len(TRAIN_IMG)), dtype=int)
        x_test = np.ndarray(shape=(len(TEST_IMG), self.img_size[0], self.img_size[1]),
                            dtype=float)
        y_test = np.ndarray(shape=(len(TEST_IMG)), dtype=int)

        # train/test set
        for img_count in range(len(TRAIN_IMG)):
            img = cv2.imread(self.train_directory + TRAIN_IMG[img_count])
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.resize( img, (self.img_size[1], self.img_size[0]))
            x_train[img_count] = img
            direction = TRAIN_IMG[img_count].split("_")[0]
            y_train[img_count] = directions_dict.get(direction)

        for img_count in range(len(TEST_IMG)):
            img = cv2.imread(self.test_directory + TEST_IMG[img_count])
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.resize( img, (self.img_size[1], self.img_size[0]))
            x_test[img_count] = img
            direction = TEST_IMG[img_count].split("_")[0]
            y_test[img_count] = directions_dict.get(direction)
        
        print("The dataset has been splitted successfully")
        return x_train, y_train, x_test, y_test



    def train_model(self, X, Y, validation_data: tuple = None, dropout: float = 0.0, early_stopping: bool = True, plot_summary: bool = False,
                    plot_history: bool = False, data_augmentation: bool = False, batch_size: int = 32,
                    epochs: int = 10, plot_confusion_matrix: bool = False):

        x_train, y_train, x_test, y_test = X, Y, validation_data[0], validation_data[1]
        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], x_train.shape[2], 1))
        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], x_test.shape[2], 1))

        model = Sequential()
        model.add(Conv2D(64, (3, 3), strides=(1, 1), padding='same', activation=relu,
                         input_shape=x_train[0].shape))
        model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
        model.add(Dropout(dropout))

        model.add(Conv2D(128, (3, 3), strides=(1, 1), padding='same', activation=relu))
        model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
        model.add(Dropout(dropout))

        model.add(Conv2D(128, (3, 3), strides=(1, 1), padding='same', activation=relu))
        #model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
        model.add(Dropout(dropout))

        model.add(Flatten())

        model.add(Dense(64, activation=relu))
        model.add(Dropout(dropout))
        model.add(Dense(len(self.directions), activation=softmax))

        if plot_summary: model.summary()

        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss="sparse_categorical_crossentropy",
            metrics=['accuracy'],
        )

        if early_stopping:
            callbacks = [EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)]
        else:
            callbacks = []

        if data_augmentation:

            datagen = ImageDataGenerator(rotation_range=20,
                                         zoom_range=0.15,
                                         width_shift_range=0.2,
                                         height_shift_range=0.2,
                                         shear_range=0.15,
                                         fill_mode="nearest")

            train_datagen = datagen.flow(x_train, y_train, batch_size=batch_size)
            test_datagen = datagen.flow(x_test, y_test, batch_size=batch_size)
            steps = int(x_train.shape[0] / batch_size)
            test_steps = int(x_test.shape[0] / batch_size)
            hist = model.fit_generator(train_datagen, epochs=epochs, steps_per_epoch=steps,
                                   callbacks=callbacks,validation_data=test_datagen, validation_steps=test_steps)
  
        else:
            hist = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size,
                                   callbacks=callbacks, validation_data=(x_test, y_test))

        if plot_history: self.__history_model(hist.history)
        try:
            model_name = self.models_directory + "model_" + str(datetime.datetime.now())[:16] + ".h5"
            model.save(model_name)
            print("model saved as '{}'".format(model_name))
        except:
            print("Error when saving the model !")
        
        if plot_confusion_matrix:
            y_pred = model.predict(x_test)
            self.__confusion(self.directions, y_test, np.argmax(y_pred, axis=1))

        return model


    def load_model(self, path_to_model: str = None, 
                        newest_model_in_directory: bool = False,
                        validation_data: tuple = None):

        if newest_model_in_directory and path_to_model is None:
            model_name = self.__newest(self.models_directory)
        elif not newest_model_in_directory and path_to_model is not None:
            model_name = path_to_model
        else:
            raise Exception('please specify either a model or a folder containing models !')

        model = load_model(model_name)
        input_shape = model.input_shape
        self.img_size = (model.input_shape[1], model.input_shape[2])

        if validation_data is not None:
            score = model.evaluate(validation_data[0], validation_data[1], verbose=0)
            print("model %s => loss: %.2f, accuracy: %.2f%%" % (model_name, score[0], score[1] * 100))
        return model

 
    def test_record(self, model, speed: int, accuracy: float, display_text=True):
        cap = cv2.VideoCapture(0)
        #time.sleep(0.5)
        counter = 0
        
        #time.sleep(5)
        class_name = None
        class_predicted_rate = 0.0

        shot = np.ndarray(shape=(1, self.img_size[0], self.img_size[1], 1), dtype=float)
        time_ = 0
        roi_hist = self.__define_mask(cv2.imread(self.mask_path), self.mask_path, resized = True)

        while True:
            ret, frame = cap.read()
            cv2.rectangle(frame, (self.mask_x, self.mask_y), (self.mask_x+self.mask_width, self.mask_y+self.mask_height), (255,0,0), 2)
            
            mask = self.__show_mask(frame, roi_hist)
            cv2.imshow("mask", mask)

            if display_text:
                if class_name is not None:
                    self.__webcam_with_text(frame, ["{}: {}%".format(class_name, class_predicted_rate*100)])
                else:
                    self.__webcam_with_text(frame, [])
            time_ += 1
            if time_ % speed == 0:
                mask = np.reshape(mask, (mask.shape[0], mask.shape[1], 1))
                shot[0] = mask
                pred = model.predict(shot)
                class_predicted_rate = np.max(pred)
                if class_predicted_rate >= accuracy:
                    pred = pred.argmax(axis=-1)[0]
                    class_name = self.directions[pred]
                else:
                    class_predicted_rate = 0.0
                    class_name = None
                print("Classe: ", class_name)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                os.remove(filename)
                break

        cap.release()
        cv2.destroyAllWindows()

    
    def cam_init(self):
        capture = cv2.VideoCapture(0)
        return capture
  


    def test_record2(self, model, speed: int, accuracy: float, display_text=True):
        cap = cv2.VideoCapture(0)
        time.sleep(2)
        counter = 0
        
        class_name = None
        class_predicted_rate = 0.0
        #pp = np.reshape(pp, (1, pp.shape[0]))
        shot = np.ndarray(shape=(1, self.img_size[0], self.img_size[1], 1), dtype=int)
        time_ = 0
        mask = self.__define_mask(cv2.imread(self.mask_path), self.mask_path, resized = True)

        while True:
            ret, frame = cap.read()
            cv2.rectangle(frame, (mask_x, mask_y), (mask_x+mask_width, mask_y+mask_height), (255,0,0), 2)
            
            mask = self.__show_mask(frame, mask)
            cv2.imshow("mask", mask)

            if display_text:
                if class_name is not None:
                    self.__webcam_with_text(frame, ["{}: {}%".format(class_name, class_predicted_rate*100)])
                else:
                    self.__webcam_with_text(frame, [])
            time_ += 1
            if time_ % speed == 0:
                mask = np.reshape(mask, (1, self.img_size[0] * self.img_size[1]))
                shot[0] = mask
                
                pred = model.predict_proba(shot)
                class_predicted_rate = np.max(pred)
                if class_predicted_rate >= accuracy:
                    pred = pred.argmax(axis=-1)[0]
                    class_name = self.directions[pred]
                else:
                    class_predicted_rate = 0.0
                    class_name = None
                print("Classe: ", class_name)

            if cv2.waitKey(1) == 27:
                os.remove(filename)
                print("Stopping webcam ...")
                break

        cap.release()
        cv2.destroyAllWindows()

    
    def cam_init(self):
        capture = cv2.VideoCapture(0)
        return capture

    def test_record2(self, capture, model, speed: int, accuracy: float, display_text=True):
        
        
        class_name = None
        class_predicted_rate = 0.0
        shot = np.ndarray(shape=(1, self.img_size[0], self.img_size[1], 3), dtype=float)

        
        time_ = 0

        while True:
            ret, frame = capture.read()
            if display_text:
                if class_name is not None:
                    self.__webcam_with_text(frame, ["{}: {}%".format(class_name, class_predicted_rate*100)])
                else:
                    self.__webcam_with_text(frame, [])
            time_ += 1
            if time_ % speed == 0:
                filename = self.working_directory + "shot_" + str(time_ % speed) + ".jpg"
                cv2.imwrite(filename, img=cv2.resize(frame, (self.img_size[1], self.img_size[0])))
                
                
                shot[0] = cv2.imread(filename)
                os.remove(filename)
                pred = model.predict(shot)
                class_predicted_rate = np.max(pred)
                if class_predicted_rate >= accuracy:
                    pred = pred.argmax(axis=-1)[0]
                    class_name = self.directions[pred]
                else:
                    class_predicted_rate = 0.0
                    class_name = None

                return class_name

            if cv2.waitKey(1) == 27:
                print("Stopping webcam ...")
                break

        
    def cam_destroy(self, capture):
        capture.release()
        cv2.destroyAllWindows()
