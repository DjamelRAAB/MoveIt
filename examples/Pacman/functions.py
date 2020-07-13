from constants import *

font = pygame.font.SysFont(None, 25)
white = (255, 255, 255)


def message_to_screen(msg, color):
    if color == "black": color_RGB = (0, 0, 0)
    if color == "red": color_RGB = (255, 0, 0)

    screen_text = font.render(msg, True, color_RGB)
    window.blit(screen_text, WINDOW_SIZE / 2)


def pause():
    paused = True

    while paused:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RETURN:
                    paused = False
                elif event.key == pygame.K_q:
                    pygame.quit()
                    quit()

        window.fill(white)
        message_to_screen("Paused", "black")
        message_to_screen("Press 'enter' to continue or Q to quit.", "black")
        pygame.display.update()
        pygame.time.Clock().tick(5)
