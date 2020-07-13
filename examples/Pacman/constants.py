"""
        _________________

        Game's parameters
        _________________
"""

import pygame
from pygame import *
from pygame.locals import *
import time
import random

# Game PARAMETERS
TICK_LIMIT = 30

# Display
SPRITE_SIZE = 30
SPRITE_HEIGHT = 22
SPRITE_WIDTH = 19
WINDOW_SIZE = (SPRITE_WIDTH * SPRITE_SIZE, SPRITE_HEIGHT * SPRITE_SIZE)
WINDOW_TITLE = "AIPAC-MAN"

ICON = "resources/pacman_right.png"


# Import resources
root = "resources/"
LEVEL_MAP = root+"map.txt"
HOME_SCREEN = root+"home.jpeg"
#HOME_SCREEN = root+"home2.png"
BACKGROUND_IMG = root+"pacman_up.png"
WALL_IMG = root+"wall.png"
WAY_IMG = root+"bck.jpg"
POINT_IMG = root+"point.png"

# Pacman direction images
PACMAN_UP = root+"pacman_up.png"
PACMAN_RIGHT = root+"pacman_right.png"
PACMAN_DOWN = root+"pacman_down.png"
PACMAN_LEFT = root+"pacman_left.png"

# Ghost color images
GHOST_RED = root+"ghost_red.png"
GHOST_YELLOW = root+"ghost_yellow.png"
GHOST_GREEN = root+"ghost_green.jpg"

