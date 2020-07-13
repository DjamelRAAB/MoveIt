"""
    ____________________________________________________
                    <<  Pac-Man game  >>
    Files : main.py, classes.py, constants.py, resources
    ____________________________________________________
"""

import cv2
import os
import time
import datetime



from classes import *


ghost_move_count = 0
pygame.init()

random.seed()
direction = ""
score = 0
x_change, y_change = 0, 0
#  Pygame window
window = pygame.display.set_mode(WINDOW_SIZE)
icon = pygame.image.load(ICON)
pygame.display.set_icon(icon)
pygame.display.set_caption(WINDOW_TITLE)

# Principal loop
game_loop = 1
while game_loop:
    # loading the home screen
    home_screen = pygame.image.load(HOME_SCREEN).convert()
    window.blit(home_screen, (120, 250))
    pygame.display.flip()  # to refresh the screen

    #  We set these variables to 1 at every turn loop
    playing_loop = 0
    home_loop = 1

    # home loop
    while home_loop:

        pygame.time.Clock().tick(TICK_LIMIT)  # to limit the loop speed

        for event in pygame.event.get():
            # If the user leaves, we set the variables to 0 and close all
            if event.type == QUIT or event.type == KEYDOWN and event.key == K_ESCAPE:
                playing_loop = 0
                home_loop = 0
                game_loop = 0
            elif event.type == KEYDOWN and event.key == K_RETURN:
                pygame.time.Clock().tick(TICK_LIMIT)  # to limit the loop speed
                # Loading the background
                background = pygame.image.load(BACKGROUND_IMG).convert()

                #  generate level
                level_generator = Level()
                # level_generator.generate()
                level_generator.display(window)
                pacman = PacMan()
                blinky = Ghosts(2, 1, "red")
                clyde = Ghosts(SPRITE_WIDTH - 2, SPRITE_HEIGHT - 2, "yellow")
                playing_loop = 1
                home_loop = 0

    # Playing the game
    while playing_loop:

        is_keydown = False
        pacman_last_pos = (pacman.case_y, pacman.case_x)
        for event in pygame.event.get():

            # If the user leaves, we set the variables to 0 and close all
            if event.type == QUIT:
                playing_loop = 0
                game_loop = 0

            elif event.type == KEYDOWN:
                is_keydown = True
                ghost_move_count = 1  #  reset to 1, to calculate new path

                # if the user presses escape button here, it only returns to the menu
                if event.key == K_ESCAPE:
                    playing_loop = 0

                # moving PACMAN  
                elif event.key == K_UP:
                    direction = "up"
                elif event.key == K_LEFT:
                    direction = "left"
                elif event.key == K_DOWN:
                    direction = "down"
                elif event.key == K_RIGHT:
                    direction = "right"
                pacman.change_direction(direction)

                pacman.move()

        if not is_keydown:
            pacman.move()
            ghost_move_count += 1
            # ghost.move()

        # Display the new positions
        window.blit(background, (0, 0))
        if level_generator.structure[pacman.case_y][pacman.case_x] == "p":
            level_generator.structure[pacman.case_y][pacman.case_x] = "a"
            score += 1

        #  if pacman's position has changed, calculate the new path
        blinky.move_Blinky(pacman.case_x, pacman.case_y)
        clyde.move_Clyde(pacman.case_x, pacman.case_y, direction)

        level_generator.display(window)
        window.blit(pacman.direction, (pacman.x + 5, pacman.y + 5))  # +5 refers to center pacman
        window.blit(blinky.img, (blinky.x + 5, blinky.y + 5))  # +5 refers to center pacman
        window.blit(clyde.img, (clyde.x + 5, clyde.y + 5))  # +5 refers to center pacman
        # window.fill((255, 0, 0), rect=(blinky.path[0][1] * 30, blinky.path[0][0] * 30, 30, 30))
        # window.fill((255,155, 0), rect=(clyde.path[0][1] * 30, clyde.path[0][0] * 30, 30, 30))
        pygame.display.flip()

        pygame.time.Clock().tick(1.5)

        # Once we win, return to home
        if (pacman.case_y, pacman.case_x) == (blinky.case_y, blinky.case_x) \
                or (pacman.case_y, pacman.case_x) == (clyde.case_y, clyde.case_x):
            print("Game over")
            playing_loop = 0
