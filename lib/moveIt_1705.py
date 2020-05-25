import cv2
import os
import time
import datetime
import numpy as np
import shutil
# import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.activations import *
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


class moveIt:

    def __init__(self, moves_list: list, img_per_direction: int = 100,
                 count_test_img: int = 20, 
                 working_directory: str = "working/", train_directory: str = "train/",
                 test_directory: str = "test/", models_directory: str = "models/"):

        if img_per_direction < count_test_img:
            raise Exception('count_test_img should not exceed img_per_direction')

        self.directions = moves_list
        self.img_size = None
        self.img_count = img_per_direction
        self.count_test_img = count_test_img
        self.webcam_image_size = None
        self.working_directory = working_directory
        self.train_directory = working_directory + train_directory
        self.test_directory = working_directory + test_directory
        self.models_directory = working_directory + models_directory

        if not os.path.exists(self.working_directory): os.makedirs(self.working_directory)
        if not os.path.exists(self.train_directory): os.makedirs(self.train_directory)
        if not os.path.exists(self.test_directory): os.makedirs(self.test_directory)
        if not os.path.exists(self.models_directory): os.makedirs(self.models_directory)
        # [ os.makedirs(self.working_directory+"/"+move+"/") for move in self.directions if not os.path.exists(self.working_directory+"/"+move+"/")  ]
        print("moveIt ready to use !")

    def __str__(self):
        return 'MoveIt class: \nmoves: %s\n image size: %s\n resized image: %s\n image per move: %s\n test image per move: %s\n' \
                 %(self.directions, str(self.img_size), str(self.webcam_image_size), self.img_count, self.count_test_img)

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

    def set_moves(self):
        # Initialize the webcam
        cap = cv2.VideoCapture(0)
        time.sleep(2)

        directions_counter = 0
        count = 0
        taking_picture = False

        while True:
            # starting the webcam
            ret, frame = cap.read()
            if not taking_picture:
                if directions_counter != 0:
                    self.__webcam_with_text(frame, ["Move {} succefully saved !".format(directions_counter),
                                                    "Move {}: press enter to start !".format(directions_counter + 1)])
                else:
                    self.__webcam_with_text(frame, ["Move {}: press enter to start !".format(directions_counter + 1)])
            else:
                self.__webcam_with_text(frame, [], display_text=False)
            key = cv2.waitKey(1)
            if key == 13:  # 13 is the Enter key
                taking_picture = True
                count = 0
                direction = self.directions[directions_counter]
                print("starting pictures taking ({}) ...".format(direction))

            if taking_picture:
                #  create the folders
                direction_folder = self.working_directory + self.directions[directions_counter]
                if not os.path.exists(direction_folder): os.makedirs(direction_folder)
                if count < self.img_count:
                    filename = direction_folder + "/" + self.directions[directions_counter] + "_" + str(count) + ".jpg"
                    cv2.imwrite(filename, img=frame)
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
                break

        # Release the webcam
        cap.release()
        cv2.destroyAllWindows()

    
    def split_data(self, img_size: tuple):
        self.img_size = img_size
        try:
            for direction in range(len(self.directions)):
                for img in range(self.img_count):
                    source_file = self.working_directory + self.directions[direction] + "/" + self.directions[
                        direction] + "_" + str(img) + ".jpg"
                    # moving to TEST folder ELSE train
                    if img % int(self.img_count / self.count_test_img) == 0:
                        dest_file = self.test_directory + self.directions[direction] + "_" + str(img) + ".jpg"
                    else:
                        dest_file = self.train_directory + self.directions[direction] + "_" + str(img) + ".jpg"
                    #shutil.move(source_file, dest_file)
                    shutil.copy2(source_file, dest_file)
        except:
            print("Warning: the files have already been moved !")

        TRAIN_IMG = os.listdir(self.train_directory)
        TEST_IMG = os.listdir(self.test_directory)
        directions_dict = dict(zip(self.directions, range(len(self.directions))))

        x_train = np.ndarray(
            shape=(len(TRAIN_IMG), self.img_size[0], self.img_size[1], 3),
            dtype=int)
        y_train = np.ndarray(shape=(len(TRAIN_IMG)), dtype=int)
        x_test = np.ndarray(shape=(len(TEST_IMG), self.img_size[0], self.img_size[1], 3),
                                 dtype=int)
        y_test = np.ndarray(shape=(len(TEST_IMG)), dtype=int)

        # train/test set
        for img_count in range(len(TRAIN_IMG)):
            x_train[img_count] = cv2.resize( cv2.imread(self.train_directory + TRAIN_IMG[img_count]), (self.img_size[1], self.img_size[0]))
            direction = TRAIN_IMG[img_count].split("_")[0]
            y_train[img_count] = directions_dict.get(direction)

        for img_count in range(len(TEST_IMG)):
            x_test[img_count] = cv2.resize( cv2.imread(self.test_directory + TEST_IMG[img_count]), (self.img_size[1], self.img_size[0]))
            direction = TEST_IMG[img_count].split("_")[0]
            y_test[img_count] = directions_dict.get(direction)

        return x_train, y_train, x_test, y_test

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

    def train_model(self, X, Y, validation_data: tuple = None, dropout: float = 0.0, early_stopping: bool = True, plot_summary: bool = False,
                    plot_history: bool = False, data_augmentation: bool = False, batch_size: int = 20,
                    epochs: int = 10, plot_confusion_matrix: bool = False):

        x_train, y_train, x_test, y_test = X, Y, validation_data[0], validation_data[1]

        model = Sequential()
        model.add(Conv2D(128, (3, 3), strides=(1, 1), padding='same', activation=relu,
                         input_shape=(self.img_size[0], self.img_size[1], 3)))
        model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
        model.add(Dropout(dropout))

        model.add(Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation=relu))
        model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
        model.add(Dropout(dropout))

        model.add(Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation=relu))
        model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
        model.add(Dropout(dropout))

        model.add(Flatten())

        model.add(Dense(120, activation=relu))
        model.add(Dropout(dropout))
        #model.add(Dense(80, activation=relu))
        model.add(Dense(len(self.directions), activation=softmax))
        print("len(self.directions)", len(self.directions))

        if plot_summary: model.summary()

        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss=SparseCategoricalCrossentropy(),
            #loss="sparse_categorical_crossentropy",
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
            #cm = confusion_matrix(y_test, np.argmax(y_pred, axis=1))
            #ax = sns.heatmap(cm, annot=True, fmt="d")
            #plt.show()
            self.__confusion(y_test, np.argmax(y_pred, axis=1))

        return model

    def __confusion(self, true_targets, predicted_targets):
    
        #cm = tensorflow.math.confusion_matrix(labels=test_label.argmax(1), predictions=predict_one_hot.argmax(1), num_classes=len(class_names))
        cm = confusion_matrix(true_targets, predicted_targets)
        fig, ax = plt.subplots(1, 1, figsize=(6, 5))
        ax = sns.heatmap(cm, annot=True, fmt="d", cmap='YlGnBu')
        plt.tight_layout()
        tick_marks = np.arange(len(self.directions))
        plt.xticks(tick_marks, self.directions)
        plt.yticks(tick_marks, self.directions)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        
        plt.show()

    
    def __newest(self, path: str):

        files = os.listdir(path)
        paths = [os.path.join(path, basename) for basename in files]
        return max(paths, key=os.path.getctime)

    def load_model(self, path_to_model: str = None, newest_model_in_directory: bool = False,
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

    def record(self, model, speed: int, accuracy: float, display_text=True):
        cap = cv2.VideoCapture(0)
        time.sleep(2)
        counter = 0
        class_name = None
        class_predicted_rate = 0.0

        shot = np.ndarray(shape=(1, self.img_size[0], self.img_size[1], 3), dtype=int)
        time_ = 0

        while True:
            ret, frame = cap.read()
            time_ += 1
            if time_ % speed == 0:
                filename = self.working_directory + "shot_" + str(time_ % speed) + ".jpg"
                cv2.imwrite(filename, img=cv2.resize(frame, (self.img_size[1], self.img_size[0])))
                shot[0] = cv2.imread(filename)
                pred = model.predict(shot)
                class_predicted_rate = np.max(pred)
                if class_predicted_rate >= accuracy:
                    pred = pred.argmax(axis=-1)[0]
                    class_name = self.directions[pred]
                    #return str(class_name)
                    return class_name
                else:
                    return None
                    
            if cv2.waitKey(1) & 0xFF == ord('q'):
                os.remove(filename)
                break

        cap.release()
        cv2.destroyAllWindows()


    def test_record(self, model, speed: int, accuracy: float, display_text=True):
        cap = cv2.VideoCapture(0)
        time.sleep(2)
        counter = 0
        
        #time.sleep(5)
        class_name = None
        class_predicted_rate = 0.0

        shot = np.ndarray(shape=(1, self.img_size[0], self.img_size[1], 3), dtype=int)
        time_ = 0

        while True:
            ret, frame = cap.read()
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
                pred = model.predict(shot)
                class_predicted_rate = np.max(pred)
                #print(class_predicted_rate*100,"%")
                if class_predicted_rate >= accuracy:
                    pred = pred.argmax(axis=-1)[0]
                    class_name = self.directions[pred]
                    #print("shot {} : {}".format(int(counter / 5), class_name))
                else:
                    class_predicted_rate = 0.0
                    class_name = None
                    #print("Try again please !")

            if cv2.waitKey(1) & 0xFF == ord('q'):
                os.remove(filename)
                break

        cap.release()
        cv2.destroyAllWindows()

    def test_record2(self, model, speed: int, accuracy: float, display_text=True):
        cap = cv2.VideoCapture(0)
        #time.sleep(2)
        counter = 0
        
        #time.sleep(5)
        class_name = None
        class_predicted_rate = 0.0
        shot = np.ndarray(shape=(1, self.img_size[0], self.img_size[1], 3), dtype=int)

        
        time_ = 0

        while True:
            ret, frame = cap.read()
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
                #print(class_predicted_rate*100,"%")
                if class_predicted_rate >= accuracy:
                    pred = pred.argmax(axis=-1)[0]
                    class_name = self.directions[pred]
                    
                    #print("shot {} : {}".format(int(counter / 5), class_name))
                else:
                    class_predicted_rate = 0.0
                    class_name = None
                    
                return class_name
                    #print("Try again please !")

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()    