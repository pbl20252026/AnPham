import pygame
import sys

# ================= LOGIC TỪ CODE GỐC =================


def create_arr(arr1, arr2, arr3):
    mergeArr = arr1 + arr2 + arr3
    new_arr1, new_arr2, new_arr3 = [], [], []
    a, b, c = 0, 1, 2
    for _ in range(5):
        new_arr1.append(mergeArr[a])
        new_arr2.append(mergeArr[b])
        new_arr3.append(mergeArr[c])
        a += 3
        b += 3
        c += 3
    return new_arr1, new_arr2, new_arr3


def find_number(choice, arr1, arr2, arr3):
    if choice == 1:
        arr1, arr2 = arr2, arr1
    elif choice == 3:
        arr3, arr2 = arr2, arr3
    return create_arr(arr1, arr2, arr3)


# ================== PYGAME SETUP ==================
pygame.init()
WIDTH, HEIGHT = 900, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("🎩 Magic Number Predictor")
clock = pygame.time.Clock()
# ================= FONT =================
# Chỉ dùng SysFont vì game đã chuyển sang tiếng Anh
font = pygame.font.SysFont(None, 32)
big_font = pygame.font.SysFont(None, 48)

# ================= GAME DATA =================


def reset_game():
    return (
        [1, 4, 7, 10, 13],
        [2, 5, 8, 11, 14],
        [3, 6, 9, 12, 15],
        0,
        None
    )


arr1, arr2, arr3, round_count, result = reset_game()

game_state = "START"  # START | PLAY | END

columns = [
    pygame.Rect(100, 150, 200, 300),
    pygame.Rect(350, 150, 200, 300),
    pygame.Rect(600, 150, 200, 300)
]

start_button = pygame.Rect(WIDTH//2 - 120, HEIGHT//2 + 40, 240, 60)

# ================= DRAW FUNCTIONS =================


def draw_column(rect, arr):
    pygame.draw.rect(screen, (200, 200, 200), rect, border_radius=10)
    for i, num in enumerate(arr):
        txt = font.render(str(num), True, (0, 0, 0))
        screen.blit(txt, (rect.x + 85, rect.y + 30 + i * 45))


def draw_center_text(text, y, big=False):
    f = big_font if big else font
    txt = f.render(text, True, (255, 255, 255))
    screen.blit(txt, (WIDTH // 2 - txt.get_width() // 2, y))


def draw_button(rect, text):
    pygame.draw.rect(screen, (70, 130, 180), rect, border_radius=12)
    txt = font.render(text, True, (255, 255, 255))
    screen.blit(txt, (rect.centerx - txt.get_width() // 2,
                      rect.centery - txt.get_height() // 2))


# ================= MAIN LOOP =================
running = True
while running:
    screen.fill((30, 30, 40))

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

        if event.type == pygame.MOUSEBUTTONDOWN:
            # START SCREEN
            if game_state == "START" and start_button.collidepoint(event.pos):
                game_state = "PLAY"

            # PLAY SCREEN
            elif game_state == "PLAY" and result is None:
                for idx, col in enumerate(columns):
                    if col.collidepoint(event.pos):
                        round_count += 1
                        arr1, arr2, arr3 = find_number(
                            idx + 1, arr1, arr2, arr3)
                        if round_count == 3:
                            result = arr2[2]
                            game_state = "END"

            # END SCREEN (click để chơi lại)
            elif game_state == "END":
                arr1, arr2, arr3, round_count, result = reset_game()
                game_state = "START"

    # ================= DRAW STATES =================
    if game_state == "START":
        draw_center_text(" MAGIC NUMBER TRICK ", 160, big=True)
        draw_center_text("Think of ONE number from 1 to 15", 230)
        draw_center_text("Keep it in your mind!", 270)
        draw_button(start_button, "START GAME")

    elif game_state == "PLAY":
        draw_center_text("Click the column that CONTAINS your number", 40)
        draw_center_text(f"Round: {round_count}/3", 80)

        draw_column(columns[0], arr1)
        draw_column(columns[1], arr2)
        draw_column(columns[2], arr3)

    elif game_state == "END":
        draw_center_text("RESULT", 180, big=True)
        draw_center_text(f"Your number is: {result}", 250)
        draw_center_text("Click anywhere to play again", 320)

    pygame.display.flip()
    clock.tick(60)
