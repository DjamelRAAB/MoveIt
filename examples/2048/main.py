import random
from tkinter import Frame, Label, CENTER
import os
import sys
import logic
import constants as c
import time

ROOT_DIR = os.path.abspath("../../")
sys.path.append(ROOT_DIR)
from moveIt import *


class GameGrid(Frame):
    def __init__(self):
        Frame.__init__(self)

        self.grid()
        self.master.title('2048')
        

        # init moveIt
        self.directions = ["up", "right", "down", "left"]
        self.moveit = MoveIt_ML(self.directions)
        #moveit.set_moves()
        x_train, y_train, x_test, y_test = self.moveit.split_data()
        try:
            self.model = self.moveit.load_model(path_to_model="working/models/knn_3n_classifier_2020-06-30 18:54.sav")
            print("model ready !")
        except:
            print("Cannot load model !")
        #direction, acc = self.moveit.test_record(self.model, 15, 0.5)

        self.commands = {c.KEY_UP: logic.up, c.KEY_DOWN: logic.down,
                         c.KEY_LEFT: logic.left, c.KEY_RIGHT: logic.right}
        
        self.grid_cells = []
        self.init_grid()
        self.init_matrix()

    def play(self):
        self.i = 0
        self.cam = self.moveit.cam_init()
        direction, acc = None, 0
        while True:
            direction, acc = self.moveit.predict(self.cam, self.model, 25, 0.5, prev_class = direction, prev_rate = acc)
            print("direction %s : %s (%.0f%%) " %(self.i, direction, acc*100) )
            if direction is not None:
                self.key_down(direction)
            self.i += 1

        self.moveit.cam_destroy(cam)


    def init_grid(self):
        background = Frame(self, bg=c.BACKGROUND_COLOR_GAME,
                           width=c.SIZE, height=c.SIZE)
        background.grid()

        for i in range(c.GRID_LEN):
            grid_row = []
            for j in range(c.GRID_LEN):
                cell = Frame(background, bg=c.BACKGROUND_COLOR_CELL_EMPTY,
                             width=c.SIZE / c.GRID_LEN,
                             height=c.SIZE / c.GRID_LEN)
                cell.grid(row=i, column=j, padx=c.GRID_PADDING,
                          pady=c.GRID_PADDING)
                t = Label(master=cell, text="",
                          bg=c.BACKGROUND_COLOR_CELL_EMPTY,
                          justify=CENTER, font=c.FONT, width=5, height=2)
                t.grid()
                grid_row.append(t)

            self.grid_cells.append(grid_row)

    def gen(self):
        return random.randint(0, c.GRID_LEN - 1)

    def init_matrix(self):
        self.matrix = logic.new_game(c.GRID_LEN)
        self.history_matrixs = list()
        self.matrix = logic.add_two(self.matrix)
        self.matrix = logic.add_two(self.matrix)

    def update_grid_cells(self):
        for i in range(c.GRID_LEN):
            for j in range(c.GRID_LEN):
                new_number = self.matrix[i][j]
                if new_number == 0:
                    self.grid_cells[i][j].configure(
                        text="", bg=c.BACKGROUND_COLOR_CELL_EMPTY)
                else:
                    self.grid_cells[i][j].configure(text=str(
                        new_number), bg=c.BACKGROUND_COLOR_DICT[new_number],
                        fg=c.CELL_COLOR_DICT[new_number])
        self.update_idletasks()

    def key_down(self, event):
        self.matrix, done = self.commands[event](self.matrix)

        state = self.game_over(self.matrix)

        if state != 'continue':
            if state == 'win':
                self.grid_cells[1][1].configure(
                    text="You", bg=c.BACKGROUND_COLOR_CELL_EMPTY)
                self.grid_cells[1][2].configure(
                    text="Win!", bg=c.BACKGROUND_COLOR_CELL_EMPTY)
                
            elif state == 'lose':
                self.grid_cells[1][1].configure(
                    text="You", bg=c.BACKGROUND_COLOR_CELL_EMPTY)
                self.grid_cells[1][2].configure(
                    text="Lose!", bg=c.BACKGROUND_COLOR_CELL_EMPTY)
            
            self.update_idletasks()
        else:
            self.matrix = logic.add_two(self.matrix)
            self.update_grid_cells()

    def generate_next(self):
        index = (self.gen(), self.gen())
        while self.matrix[index[0]][index[1]] != 0:
            index = (self.gen(), self.gen())
        self.matrix[index[0]][index[1]] = 2

    
    def game_over(self, current_matrix):
        
        if 2048 in np.array(current_matrix):
            return "win"
        for direction in self.directions:
            if self.commands[direction](current_matrix) != current_matrix:
                return "continue"
        
        return "lose"
        

gamegrid = GameGrid()
gamegrid.play()