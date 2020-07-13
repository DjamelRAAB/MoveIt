"""
        __________________________________

        Game's classes : Level, Personnage
        __________________________________
"""

"""
    Class for creating the levels
"""

from constants import *


class Level:

    def __init__(self):
        self.file = LEVEL_MAP
        self.structure = []

    def generate(self):
        """ method used to generate level depending on the file."""

        if self.structure == []:
            with open(self.file, "r") as file:

                level_structure = []

                for line in file:
                    lines = []
                    for sprite in line:
                        if sprite != '\n':
                            lines.append(sprite)
                    level_structure.append(lines)
                self.structure = level_structure

    def display(self, window):
        """Method for displaying the level depending on
        the structure of list returned by generate() method"""

        self.generate()
        wall = pygame.image.load(WALL_IMG).convert()
        way = pygame.image.load(WAY_IMG).convert()
        point = pygame.transform.scale(pygame.image.load(POINT_IMG), (5, 5))

        # browse throught the level lines
        line_num = 0
        for line in self.structure:
            square_num = 0
            for sprite in line:
                x = square_num * SPRITE_SIZE
                y = line_num * SPRITE_SIZE
                if sprite == 'w':  # w = wall
                    window.blit(wall, (x, y))
                elif sprite == 'p':
                    window.blit(way, (x, y))
                    window.blit(point, (x + 12, y + 12))
                else:
                    window.blit(way, (x, y))
                square_num += 1
            line_num += 1


class PacMan:

    def __init__(self):
        self.right = pygame.image.load(PACMAN_RIGHT).convert_alpha()
        self.left = pygame.image.load(PACMAN_LEFT).convert_alpha()
        self.up = pygame.image.load(PACMAN_UP).convert_alpha()
        self.down = pygame.image.load(PACMAN_DOWN).convert_alpha()
        #  Pacman position
        self.case_x = 9
        self.case_y = 8
        self.x = self.case_x * SPRITE_SIZE
        self.y = self.case_y * SPRITE_SIZE

        # default direction
        self.direction = self.right
        # selected level
        level_class = Level()
        level_class.generate()
        self.map = level_class.structure
        self.x_change = 0
        self.y_change = 0

    def change_direction(self, direction):

        self.x_change = 0
        self.y_change = 0

        if direction == "right":
            self.x_change = 1
            self.y_change = 0
            self.direction = self.right

        elif direction == "left":
            self.x_change = -1  # moving by one case
            self.y_change = 0  # moving by one case
            self.direction = self.left

        elif direction == "up":
            self.x_change = 0
            self.y_change = -1
            self.direction = self.up

        elif direction == "down":
            self.x_change = 0  # moving by one case
            self.y_change = 1  # moving by one case
            self.direction = self.down

    def move(self):

        if self.case_x < (SPRITE_WIDTH + self.x_change) and self.case_y < (SPRITE_HEIGHT + self.y_change):
            # check if there is not a wall
            if self.map[self.case_y + self.y_change][self.case_x + self.x_change] != 'w':
                self.case_x = self.case_x + self.x_change  # moving by one case
                self.x = self.case_x * SPRITE_SIZE
                self.case_y = self.case_y + self.y_change  # moving by one case
                self.y = self.case_y * SPRITE_SIZE


class Ghosts:  #  inherit from PacMan class

    def __init__(self, x_pos: int, y_pos: int, color: str):

        if color == "red": img = GHOST_RED
        if color == "yellow": img = GHOST_YELLOW
        if color == "green": img = GHOST_GREEN
        self.img = pygame.transform.scale(pygame.image.load(img),(20, 20)).convert_alpha()

        # self.right = pygame.transform.scale(pygame.image.load(img), (20, 20)).convert_alpha()
        # self.left = pygame.transform.scale(pygame.image.load(img), (20, 20)).convert_alpha()
        # self.up = pygame.transform.scale(pygame.image.load(img), (20, 20)).convert_alpha()
        # self.down = pygame.transform.scale(pygame.image.load(img), (20, 20)).convert_alpha()
        #  Pacman position
        # self.direction = self.right
        self.case_x = x_pos
        self.case_y = y_pos
        self.x = self.case_x * SPRITE_SIZE
        self.y = self.case_y * SPRITE_SIZE

        # selected level
        level_class = Level()
        level_class.generate()
        self.map = level_class.structure
        self.x_change = 0
        self.y_change = 0

        self.path = []

    def find_path(self, end_x: int, end_y: int):
        current_position = (0,)  # initialize tuple (y,x)
        frontier_list, visited_list = [], []
        backtracking_list = {}
        shortest_path = []
        dict_index = 0
        end = (end_y, end_x)

        frontier_list.append((self.case_y, self.case_x))

        while frontier_list != []:
            current_position = frontier_list[0]
            self.case_y, self.case_x = frontier_list[0][0], frontier_list[0][1]

            # check if there is wall or a visited cell at each side
            left_cell = (current_position[0], current_position[1] - 1)
            right_cell = (current_position[0], current_position[1] + 1)
            up_cell = (current_position[0] - 1, current_position[1])
            down_cell = (current_position[0] + 1, current_position[1])

            # left side
            if self.map[left_cell[0]][left_cell[1]] != 'w' \
                    and left_cell not in visited_list:
                frontier_list.append(left_cell)
                backtracking_list[dict_index] = (left_cell, current_position)  # (current), (previous)
                dict_index += 1

            # right side
            if self.map[right_cell[0]][right_cell[1]] != 'w' \
                    and right_cell not in visited_list:
                frontier_list.append(right_cell)
                backtracking_list[dict_index] = (right_cell, current_position)  # (current), (previous)
                dict_index += 1

            # up side
            if self.map[up_cell[0]][up_cell[1]] != 'w' \
                    and up_cell not in visited_list:
                frontier_list.append(up_cell)
                backtracking_list[dict_index] = (up_cell, current_position)  # (current), (previous)
                dict_index += 1

            # down side
            if self.map[down_cell[0]][down_cell[1]] != 'w' \
                    and down_cell not in visited_list:
                frontier_list.append(down_cell)
                backtracking_list[dict_index] = (down_cell, current_position)  # (current), (previous)
                dict_index += 1

            # place the current cell in the visited list
            visited_list.append(current_position)
            #  de-queue frontier list
            frontier_list.remove(current_position)

        shortest_path.append(end)
        temporary_path = end

        for i in range(len(backtracking_list) - 1, -1, -1):
            if backtracking_list[i][0] == temporary_path:
                temporary_path = backtracking_list[i][1]
                shortest_path.append(temporary_path)

        self.path = shortest_path

    def move_ghost(self):

        if len(self.path) > 1:
            # print(len(self.path))
            move = self.path[-2]  # path is an reversed list [end, n-1, ... start]
            self.y_change = move[0] - self.case_y
            self.x_change = move[1] - self.case_x

            self.case_x = self.case_x + self.x_change  # moving by one case
            self.x = self.case_x * SPRITE_SIZE
            self.case_y = self.case_y + self.y_change  # moving by one case
            self.y = self.case_y * SPRITE_SIZE

    def move_Blinky(self, pacman_case_x: int, pacman_case_y: int):
        self.find_path(pacman_case_x, pacman_case_y)
        self.move_ghost()

    def move_Clyde(self, pacman_case_x: int, pacman_case_y: int, pacman_direction: str):
        x_gap, y_gap = 0, 0
        if pacman_direction == "up":
            y_gap = -1
        elif pacman_direction == "down":
            y_gap = 1
        elif pacman_direction == "left":
            x_gap = -1
        elif pacman_direction == "right":
            x_gap = 1

        move_gap = 3

        for gap in range(move_gap, 0, -1):

            case_plus_gap_y = pacman_case_y + (gap * y_gap)
            case_plus_gap_x = pacman_case_x + (gap * x_gap)

            if 0 < case_plus_gap_y < SPRITE_HEIGHT and 0 < case_plus_gap_x < SPRITE_WIDTH \
                    and self.map[case_plus_gap_y][case_plus_gap_x] != 'w':
                self.find_path(case_plus_gap_x, case_plus_gap_y)
                break

        self.move_ghost()
