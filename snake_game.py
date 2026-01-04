import pygame
import cv2
import mediapipe as mp
import numpy as np
import random

# --- 1. SETTINGS & GRID ---
WIDTH, HEIGHT = 800, 600
SNAKE_SIZE = 20
FPS = 5 

pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Hand-Controlled Snake - Score System")
font = pygame.font.SysFont("Arial", 24, bold=True)
clock = pygame.time.Clock()

# MediaPipe Setup with Red Styling
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

# BGR for Red is (0, 0, 255)
hand_landmark_style = mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=3)
hand_connection_style = mp_drawing.DrawingSpec(color=(50, 50, 150), thickness=2)

cap = cv2.VideoCapture(0)

# --- 2. GAME VARIABLES ---
snake = [[200, 200], [180, 200], [160, 200]]
direction = "RIGHT"
food = [random.randrange(0, WIDTH, SNAKE_SIZE), random.randrange(0, HEIGHT, SNAKE_SIZE)]
score = 0
high_score = 0

def get_finger_pos(hand_landmarks):
    itip = hand_landmarks.landmark[8]
    return int(itip.x * WIDTH), int(itip.y * HEIGHT)

# --- 3. MAIN LOOP ---
running = True
while running:
    screen.fill((20, 20, 20)) 
    
    for event in pygame.event.get():
        if event.type == pygame.QUIT: running = False

    success, frame = cap.read()
    if not success: break
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        hand = results.multi_hand_landmarks[0]
        mp_drawing.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS, hand_landmark_style, hand_connection_style)

        fx, fy = get_finger_pos(hand)
        hx, hy = snake[0]

        diff_x = fx - hx
        diff_y = fy - hy
        if abs(diff_x) > abs(diff_y):
            if diff_x > 30 and direction != "LEFT": direction = "RIGHT"
            elif diff_x < -30 and direction != "RIGHT": direction = "LEFT"
        else:
            if diff_y > 30 and direction != "UP": direction = "DOWN"
            elif diff_y < -30 and direction != "DOWN": direction = "UP"

    # --- MOVE SNAKE ---
    new_head = list(snake[0])
    if direction == "RIGHT": new_head[0] += SNAKE_SIZE
    elif direction == "LEFT": new_head[0] -= SNAKE_SIZE
    elif direction == "UP":   new_head[1] -= SNAKE_SIZE
    elif direction == "DOWN": new_head[1] += SNAKE_SIZE
    snake.insert(0, new_head)

    # --- FOOD & SCORING ---
    if new_head[0] == food[0] and new_head[1] == food[1]:
        score += 10
        if score > high_score: high_score = score
        food = [random.randrange(0, WIDTH, SNAKE_SIZE), random.randrange(0, HEIGHT, SNAKE_SIZE)]
    else:
        snake.pop()

    # --- COLLISION (Reset) ---
    if (new_head[0] < 0 or new_head[0] >= WIDTH or 
        new_head[1] < 0 or new_head[1] >= HEIGHT or 
        new_head in snake[1:]):
        snake = [[200, 200], [180, 200], [160, 200]]
        direction = "RIGHT"
        score = 0

    # --- DRAWING ---
    # Draw Food
    pygame.draw.rect(screen, (255, 0, 0), (food[0], food[1], SNAKE_SIZE-2, SNAKE_SIZE-2))
    # Draw Snake
    for i, seg in enumerate(snake):
        color = (0, 255, 0) if i == 0 else (0, 180, 0)
        pygame.draw.rect(screen, color, (seg[0], seg[1], SNAKE_SIZE-2, SNAKE_SIZE-2))

    # --- RENDER SCORE UI ---
    score_txt = font.render(f"Score: {score}", True, (255, 255, 255))
    high_txt = font.render(f"High Score: {high_score}", True, (200, 200, 200))
    screen.blit(score_txt, (10, 10))
    screen.blit(high_txt, (10, 40))

    cv2.imshow("Hand Control (Red Marks)", frame)
    pygame.display.flip()
    
    if cv2.waitKey(1) & 0xFF == ord('q'): running = False
    clock.tick(FPS)

cap.release()
cv2.destroyAllWindows()
pygame.quit()