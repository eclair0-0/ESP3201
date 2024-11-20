import pygame
import sys
import math
import time 

# Pygame initialization
pygame.init()
WIDTH, HEIGHT = 600, 600
SCREEN = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Ultimate Tic-Tac-Toe")
FONT = pygame.font.Font(None, 24)  

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
LINE_COLOR = (50, 50, 50)
X_COLOR = (242, 85, 96)
O_COLOR = (28, 170, 156)
HIGHLIGHT_COLOR = (0, 255, 0)  # Green color for highlighting active subgrid

# Dimensions
CELL_SIZE = WIDTH // 9
SUB_BOARD_SIZE = WIDTH // 3
LINE_WIDTH = 3

# Game variables
board = [[[None for _ in range(3)] for _ in range(3)] for _ in range(9)]
current_player = "Player"
target_sub_board = None
game_over = False
depth = 4  # Depth for Minimax search
subgrid_wins = [None] * 9  # None means no winner yet for each subgrid
cell_scores = {}  # Dictionary to store scores for each cell during AI evaluation

# Drawing functions
def draw_lines():
    SCREEN.fill(WHITE)
    # Draw major grid for main board
    for i in range(1, 3):
        pygame.draw.line(SCREEN, BLACK, (i * SUB_BOARD_SIZE, 0), (i * SUB_BOARD_SIZE, HEIGHT), LINE_WIDTH)
        pygame.draw.line(SCREEN, BLACK, (0, i * SUB_BOARD_SIZE), (WIDTH, i * SUB_BOARD_SIZE), LINE_WIDTH)
    
    # Draw minor grid for sub-boards
    for x in range(3):
        for y in range(3):
            for i in range(1, 3):
                pygame.draw.line(SCREEN, LINE_COLOR,
                                 (x * SUB_BOARD_SIZE + i * CELL_SIZE, y * SUB_BOARD_SIZE),
                                 (x * SUB_BOARD_SIZE + i * CELL_SIZE, y * SUB_BOARD_SIZE + SUB_BOARD_SIZE), 1)
                pygame.draw.line(SCREEN, LINE_COLOR,
                                 (x * SUB_BOARD_SIZE, y * SUB_BOARD_SIZE + i * CELL_SIZE),
                                 (x * SUB_BOARD_SIZE + SUB_BOARD_SIZE, y * SUB_BOARD_SIZE + i * CELL_SIZE), 1)
    
    # Highlight the active subgrid
    if target_sub_board is not None:
        active_x = (target_sub_board % 3) * SUB_BOARD_SIZE
        active_y = (target_sub_board // 3) * SUB_BOARD_SIZE
        pygame.draw.rect(SCREEN, HIGHLIGHT_COLOR, (active_x, active_y, SUB_BOARD_SIZE, SUB_BOARD_SIZE), 5)  # 5 is the border width

def draw_symbols():
    for sb in range(9):
        winner = subgrid_wins[sb]
        pos_x = (sb % 3) * SUB_BOARD_SIZE + SUB_BOARD_SIZE // 2
        pos_y = (sb // 3) * SUB_BOARD_SIZE + SUB_BOARD_SIZE // 2

        if winner == "Player":
            pygame.draw.line(SCREEN, X_COLOR, (pos_x - 60, pos_y - 60), (pos_x + 60, pos_y + 60), 6)
            pygame.draw.line(SCREEN, X_COLOR, (pos_x + 60, pos_y - 60), (pos_x - 60, pos_y + 60), 6)
        elif winner == "AI":
            pygame.draw.circle(SCREEN, O_COLOR, (pos_x, pos_y), 60, 6)
        elif winner == "Draw":
            pygame.draw.line(SCREEN, LINE_COLOR, ((sb % 3) * SUB_BOARD_SIZE, (sb // 3) * SUB_BOARD_SIZE),
                             ((sb % 3 + 1) * SUB_BOARD_SIZE, (sb // 3 + 1) * SUB_BOARD_SIZE), 6)
            pygame.draw.line(SCREEN, LINE_COLOR, ((sb % 3 + 1) * SUB_BOARD_SIZE, (sb // 3) * SUB_BOARD_SIZE),
                             ((sb % 3) * SUB_BOARD_SIZE, (sb // 3 + 1) * SUB_BOARD_SIZE), 6)
        else:
            for r in range(3):
                for c in range(3):
                    symbol = board[sb][r][c]
                    cell_x = (sb % 3) * SUB_BOARD_SIZE + c * CELL_SIZE + CELL_SIZE // 2
                    cell_y = (sb // 3) * SUB_BOARD_SIZE + r * CELL_SIZE + CELL_SIZE // 2
                    if symbol == "Player":
                        pygame.draw.line(SCREEN, X_COLOR, (cell_x - 20, cell_y - 20), (cell_x + 20, cell_y + 20), 3)
                        pygame.draw.line(SCREEN, X_COLOR, (cell_x + 20, cell_y - 20), (cell_x - 20, cell_y + 20), 3)
                    elif symbol == "AI":
                        pygame.draw.circle(SCREEN, O_COLOR, (cell_x, cell_y), 20, 3)

def get_legal_moves(board, subgrid_wins, target_sub_board):
    moves = []
    for sb in range(9):
        # Skip subgrid if it is already won or drawn
        if subgrid_wins[sb] is not None:
            continue
        if target_sub_board is None or sb == target_sub_board:
            for r in range(3):
                for c in range(3):
                    if board[sb][r][c] is None:
                        moves.append((sb, r, c))
    return moves

def is_terminal(subgrid_wins):
    winner = check_main_grid_win(subgrid_wins)
    if winner is not None:
        return True
    return all(result is not None for result in subgrid_wins)

def evaluate(board, subgrid_wins):
    # Evaluate the main grid
    score = 0

    # Check for main grid win
    winner = check_main_grid_win(subgrid_wins)
    if winner == "AI":
        return 10000  # High positive score for AI win
    elif winner == "Player":
        return -10000  # High negative score for Player win
    elif winner == "Draw":
        return 0  # Neutral score for draw

    # Evaluate potential in main grid
    lines = [
        [subgrid_wins[0], subgrid_wins[1], subgrid_wins[2]],  # Row 0
        [subgrid_wins[3], subgrid_wins[4], subgrid_wins[5]],  # Row 1
        [subgrid_wins[6], subgrid_wins[7], subgrid_wins[8]],  # Row 2
        [subgrid_wins[0], subgrid_wins[3], subgrid_wins[6]],  # Column 0
        [subgrid_wins[1], subgrid_wins[4], subgrid_wins[7]],  # Column 1
        [subgrid_wins[2], subgrid_wins[5], subgrid_wins[8]],  # Column 2
        [subgrid_wins[0], subgrid_wins[4], subgrid_wins[8]],  # Diagonal
        [subgrid_wins[2], subgrid_wins[4], subgrid_wins[6]],  # Diagonal
    ]

    for line in lines:
        ai_count = line.count("AI")
        player_count = line.count("Player")
        empty_count = line.count(None) + line.count("Draw")
        if ai_count > 0 and player_count == 0:
            if ai_count == 2 and empty_count == 1:
                score += 500
            elif ai_count == 1 and empty_count == 2:
                score += 100
        elif player_count > 0 and ai_count == 0:
            if player_count == 2 and empty_count == 1:
                score -= 500
            elif player_count == 1 and empty_count == 2:
                score -= 100

    # Evaluate subgrids that are not yet won
    for sb in range(9):
        if subgrid_wins[sb] is None:
            # Evaluate potential in subgrid
            sub_board = board[sb]
            sub_score = evaluate_subgrid(sub_board)
            score += sub_score

    return score

def evaluate_subgrid(sub_board):
    # Similar to evaluating a tic-tac-toe board
    score = 0
    lines = [
        [sub_board[0][0], sub_board[0][1], sub_board[0][2]],  # Row 0
        [sub_board[1][0], sub_board[1][1], sub_board[1][2]],  # Row 1
        [sub_board[2][0], sub_board[2][1], sub_board[2][2]],  # Row 2
        [sub_board[0][0], sub_board[1][0], sub_board[2][0]],  # Column 0
        [sub_board[0][1], sub_board[1][1], sub_board[2][1]],  # Column 1
        [sub_board[0][2], sub_board[1][2], sub_board[2][2]],  # Column 2
        [sub_board[0][0], sub_board[1][1], sub_board[2][2]],  # Diagonal
        [sub_board[0][2], sub_board[1][1], sub_board[2][0]],  # Diagonal
    ]

    for line in lines:
        ai_count = line.count("AI")
        player_count = line.count("Player")
        empty_count = line.count(None)
        if ai_count > 0 and player_count == 0:
            if ai_count == 2 and empty_count == 1:
                score += 10
            elif ai_count == 1 and empty_count == 2:
                score += 1
        elif player_count > 0 and ai_count == 0:
            if player_count == 2 and empty_count == 1:
                score -= 10
            elif player_count == 1 and empty_count == 2:
                score -= 1
    return score

def minimax(board, subgrid_wins, target_sub_board, depth, is_maximizing, alpha, beta):
    if is_terminal(subgrid_wins) or depth == 0:
        return evaluate(board, subgrid_wins)

    if is_maximizing:
        max_eval = -math.inf
        for move in order_moves(get_legal_moves(board, subgrid_wins, target_sub_board)):
            sb, r, c = move
            # Save state
            prev_cell = board[sb][r][c]
            prev_subgrid_win = subgrid_wins[sb]
            prev_target_sub_board = target_sub_board

            # Make move
            board[sb][r][c] = "AI"
            winner = check_win(board[sb])
            if winner:
                subgrid_wins[sb] = winner
            elif is_draw(board[sb]):
                subgrid_wins[sb] = "Draw"

            target_sub_board = (r * 3 + c) % 9
            if subgrid_wins[target_sub_board] is not None:
                target_sub_board = None

            eval = minimax(board, subgrid_wins, target_sub_board, depth - 1, False, alpha, beta)

            # Undo move
            board[sb][r][c] = prev_cell
            subgrid_wins[sb] = prev_subgrid_win
            target_sub_board = prev_target_sub_board

            max_eval = max(max_eval, eval)
            alpha = max(alpha, eval)
            if beta <= alpha:
                break
        return max_eval
    else:
        min_eval = math.inf
        for move in order_moves(get_legal_moves(board, subgrid_wins, target_sub_board)):
            sb, r, c = move
            # Save state
            prev_cell = board[sb][r][c]
            prev_subgrid_win = subgrid_wins[sb]
            prev_target_sub_board = target_sub_board

            # Make move
            board[sb][r][c] = "Player"
            winner = check_win(board[sb])
            if winner:
                subgrid_wins[sb] = winner
            elif is_draw(board[sb]):
                subgrid_wins[sb] = "Draw"

            target_sub_board = (r * 3 + c) % 9
            if subgrid_wins[target_sub_board] is not None:
                target_sub_board = None

            eval = minimax(board, subgrid_wins, target_sub_board, depth - 1, True, alpha, beta)

            # Undo move
            board[sb][r][c] = prev_cell
            subgrid_wins[sb] = prev_subgrid_win
            target_sub_board = prev_target_sub_board

            min_eval = min(min_eval, eval)
            beta = min(beta, eval)
            if beta <= alpha:
                break
        return min_eval

def ai_move():
    global board, target_sub_board, cell_scores, subgrid_wins
    best_score = -math.inf
    best_move = None
    cell_scores.clear()  # Clear previous scores

    for move in get_legal_moves(board, subgrid_wins, target_sub_board):
        sb, r, c = move
        # Save state
        prev_cell = board[sb][r][c]
        prev_subgrid_win = subgrid_wins[sb]
        prev_target_sub_board = target_sub_board

        # Make move
        board[sb][r][c] = "AI"
        winner = check_win(board[sb])
        if winner:
            subgrid_wins[sb] = winner
        elif is_draw(board[sb]):
            subgrid_wins[sb] = "Draw"

        target_sub_board = (r * 3 + c) % 9
        if subgrid_wins[target_sub_board] is not None:
            target_sub_board = None

        # Simulate the move and calculate the minimax score
        score = minimax(board, subgrid_wins, target_sub_board, depth - 1, False, -math.inf, math.inf)

        # Undo move
        board[sb][r][c] = prev_cell
        subgrid_wins[sb] = prev_subgrid_win
        target_sub_board = prev_target_sub_board

        cell_scores[(sb, r, c)] = score  # Store score for display

        if score > best_score:
            best_score = score
            best_move = move

    if best_move:
        sb, r, c = best_move
        board[sb][r][c] = "AI"

        winner = check_win(board[sb])
        if winner:
            subgrid_wins[sb] = winner
        elif is_draw(board[sb]):
            subgrid_wins[sb] = "Draw"

        target_sub_board = (r * 3 + c) % 9
        if subgrid_wins[target_sub_board] is not None:
            target_sub_board = None

# Update check_win to mark subgrid win
def check_win(sub_board):
    # Check rows, columns, and diagonals in a 3x3 sub-board
    for row in range(3):
        if sub_board[row][0] == sub_board[row][1] == sub_board[row][2] and sub_board[row][0] is not None:
            return sub_board[row][0]
    for col in range(3):
        if sub_board[0][col] == sub_board[1][col] == sub_board[2][col] and sub_board[0][col] is not None:
            return sub_board[0][col]
    if sub_board[0][0] == sub_board[1][1] == sub_board[2][2] and sub_board[0][0] is not None:
        return sub_board[0][0]
    if sub_board[0][2] == sub_board[1][1] == sub_board[2][0] and sub_board[0][2] is not None:
        return sub_board[0][2]
    return None

def is_draw(sub_board):
    return all(cell is not None for row in sub_board for cell in row) and check_win(sub_board) is None

def handle_click(pos):
    global current_player, target_sub_board
    x, y = pos
    sb = (y // SUB_BOARD_SIZE) * 3 + (x // SUB_BOARD_SIZE)
    r = (y % SUB_BOARD_SIZE) // CELL_SIZE
    c = (x % SUB_BOARD_SIZE) // CELL_SIZE

    # Ensure the selected cell is empty and the move is allowed within the current target subgrid
    if board[sb][r][c] is None and (target_sub_board is None or sb == target_sub_board) and subgrid_wins[sb] is None:
        board[sb][r][c] = "Player"

        # Check if the current subgrid is won or ends in a draw after this move
        winner = check_win(board[sb])
        if winner:
            subgrid_wins[sb] = winner  # Mark the subgrid as won by the player
        elif is_draw(board[sb]):
            subgrid_wins[sb] = "Draw"  # Mark the subgrid as drawn

        # Set the target subgrid for the AI's next move
        target_sub_board = (r * 3 + c) % 9

        # Only allow any move if the new target subgrid is already won or drawn
        if subgrid_wins[target_sub_board] is not None:
            target_sub_board = None

        current_player = "AI"  # Pass turn to AI

def check_main_grid_win(subgrid_wins):
    # Check if any player has won in the main grid
    lines = [
        [subgrid_wins[0], subgrid_wins[1], subgrid_wins[2]],  # Row 0
        [subgrid_wins[3], subgrid_wins[4], subgrid_wins[5]],  # Row 1
        [subgrid_wins[6], subgrid_wins[7], subgrid_wins[8]],  # Row 2
        [subgrid_wins[0], subgrid_wins[3], subgrid_wins[6]],  # Column 0
        [subgrid_wins[1], subgrid_wins[4], subgrid_wins[7]],  # Column 1
        [subgrid_wins[2], subgrid_wins[5], subgrid_wins[8]],  # Column 2
        [subgrid_wins[0], subgrid_wins[4], subgrid_wins[8]],  # Diagonal
        [subgrid_wins[2], subgrid_wins[4], subgrid_wins[6]],  # Diagonal
    ]
    for line in lines:
        if line[0] == line[1] == line[2] and line[0] is not None and line[0] != "Draw":
            return line[0]
    if all(result is not None for result in subgrid_wins):
        return "Draw"
    return None

def order_moves(moves):
    # Prioritize center, corners, and then edges
    center = [(1, 1)]
    corners = [(0, 0), (0, 2), (2, 0), (2, 2)]
    edges = [(0, 1), (1, 0), (1, 2), (2, 1)]

    def move_priority(move):
        _, r, c = move
        if (r, c) in center:
            return 0
        elif (r, c) in corners:
            return 1
        else:
            return 2

    return sorted(moves, key=move_priority)

# Main game loop
while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
        elif event.type == pygame.MOUSEBUTTONDOWN and current_player == "Player":
            handle_click(event.pos)

    if current_player == "AI" and not game_over:
        ai_move()
        current_player = "Player"

    # Draw the current board state
    draw_lines()
    draw_symbols()
    pygame.display.update()

    # Check for a win or draw in the main grid
    winner = check_main_grid_win(subgrid_wins)
    if winner:
        game_over = True
        print(f"Game Over! Result: {winner}")
        
        # Update the display to show the final move
        draw_lines()
        draw_symbols()
        pygame.display.update()
        
        # Pause for 3 seconds before closing the game
        time.sleep(3)
        pygame.quit()
        sys.exit()

    pygame.display.update()
