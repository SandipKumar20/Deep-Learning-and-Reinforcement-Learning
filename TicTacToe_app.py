import streamlit as st
import torch
import torch.nn as nn
import numpy as np

# Neural Network Definition 
class TicTacToeTrainer(nn.Module):
    def __init__(self):
        super(TicTacToeTrainer, self).__init__()
        
        self.fc1 = nn.Linear(10, 32)
        self.fc2 = nn.Linear(32, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 10)  # logits as output
        
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        
        return x

# Initialize the model
@st.cache_resource
def load_model():
    model = TicTacToeTrainer()
    try:
        # Try to load the saved weights
        model.load_state_dict(torch.load("TicTacToe_Weights.pth", map_location='cpu'))
        model.eval()
        return model, True
    except FileNotFoundError:
        st.warning("Model weights file 'TicTacToe_Weights.pth' not found. Using untrained model.")
        model.eval()
        return model, False

# Game logic functions
def check_winner(board):
    """Check if there's a winner on the board"""
    # Convert board to numpy array for easier checking
    b = np.array(board).reshape(3, 3)
    
    # Check rows
    for row in b:
        if row[0] == row[1] == row[2] != '':
            return row[0]
    
    # Check columns
    for col in range(3):
        if b[0][col] == b[1][col] == b[2][col] != '':
            return b[0][col]
    
    # Check diagonals
    if b[0][0] == b[1][1] == b[2][2] != '':
        return b[0][0]
    if b[0][2] == b[1][1] == b[2][0] != '':
        return b[0][2]
    
    return None

def is_board_full(board):
    """Check if the board is full"""
    return all(cell != '' for cell in board)

def board_to_neural_input(board, current_player):
    """Convert board state to neural network input format"""
    # Create 10-dimensional input vector
    neural_input = []
    
    # First 9 components: board positions
    for cell in board:
        if cell == 'X':
            neural_input.append(1)
        elif cell == 'O':
            neural_input.append(0)
        else:  # empty cell
            neural_input.append(-1)
    
    # 10th component: player/game state
    # Check if game is over (someone won or board is full)
    winner = check_winner(board)
    board_full = is_board_full(board)
    
    if winner or board_full:
        neural_input.append(-1)  # No next move possible
    else:
        neural_input.append(1 if current_player == 'X' else 0)  # Current player's turn
    
    return torch.tensor(neural_input, dtype=torch.float32).unsqueeze(0)

def get_ai_move(model, board, ai_player):
    """Get AI move using the neural network"""
    neural_input = board_to_neural_input(board, ai_player)
    
    with torch.no_grad():
        outputs = model(neural_input)
        probabilities = torch.softmax(outputs, dim=1)
        
        # Get the predicted move
        predicted_move = torch.argmax(probabilities, dim=1).item()
        
        # If predicted_move is 0, it means no move is possible
        if predicted_move == 0:
            # Fallback to random valid move
            valid_moves = [i for i, cell in enumerate(board) if cell == '']
            if valid_moves:
                return np.random.choice(valid_moves)
            return None
        
        # Convert to board position (subtract 1 because positions 1-9 in output correspond to board indices 0-8)
        board_position = predicted_move - 1
        
        # Check if the predicted position is valid (empty)
        if 0 <= board_position < 9 and board[board_position] == '':
            return board_position
        
        # If predicted position is invalid, choose a random valid move
        valid_moves = [i for i, cell in enumerate(board) if cell == '']
        if valid_moves:
            return np.random.choice(valid_moves)
        
        return None

# Streamlit App
def main():
    st.title("🎮 Tic-Tac-Toe vs AI")
    st.markdown("Play against an AI trained with neural networks!")
    
    # Load model
    model, model_loaded = load_model()
    
    if model_loaded:
        st.success("✅ Neural network model loaded successfully!")
    else:
        st.error("❌ Could not load trained model. Make sure 'TicTacToe_Weights.pth' is in the same directory.")
    
    # Initialize session state
    if 'board' not in st.session_state:
        st.session_state.board = [''] * 9
        st.session_state.current_player = 'X'  # Human player starts
        st.session_state.game_over = False
        st.session_state.winner = None
        st.session_state.human_player = 'X'
        st.session_state.ai_player = 'O'
    
    # Game settings
    st.sidebar.header("Game Settings")
    
    # Player selection
    player_choice = st.sidebar.radio("Choose your symbol:", ['X', 'O'], 
                                    index=0 if st.session_state.human_player == 'X' else 1)
    
    if player_choice != st.session_state.human_player:
        st.session_state.human_player = player_choice
        st.session_state.ai_player = 'O' if player_choice == 'X' else 'X'
        # Reset game when player changes
        st.session_state.board = [''] * 9
        st.session_state.current_player = 'X'
        st.session_state.game_over = False
        st.session_state.winner = None
    
    # Reset game button
    if st.sidebar.button("🔄 New Game"):
        st.session_state.board = [''] * 9
        st.session_state.current_player = 'X'
        st.session_state.game_over = False
        st.session_state.winner = None
    
    # Display game board
    st.markdown("### Game Board")
    
    # Create 3x3 grid
    cols = st.columns(3)
    
    for i in range(9):
        row = i // 3
        col = i % 3
        
        with cols[col]:
            if st.session_state.board[i] == '':
                # Empty cell - show button if it's human's turn and game is not over
                if (not st.session_state.game_over and 
                    st.session_state.current_player == st.session_state.human_player):
                    if st.button(f"➕", key=f"cell_{i}", help=f"Click to place {st.session_state.human_player}"):
                        # Human makes move
                        st.session_state.board[i] = st.session_state.human_player
                        st.session_state.current_player = st.session_state.ai_player
                        st.rerun()
                else:
                    st.button("⬜", key=f"empty_{i}", disabled=True)
            else:
                # Cell has X or O
                symbol = "❌" if st.session_state.board[i] == 'X' else "⭕"
                st.button(symbol, key=f"filled_{i}", disabled=True)
    
    # Check for winner or draw
    winner = check_winner(st.session_state.board)
    if winner:
        st.session_state.winner = winner
        st.session_state.game_over = True
    elif is_board_full(st.session_state.board):
        st.session_state.game_over = True
        st.session_state.winner = "Draw"
    
    # Display game status
    if st.session_state.game_over:
        if st.session_state.winner == "Draw":
            st.info("🤝 It's a draw!")
        elif st.session_state.winner == st.session_state.human_player:
            st.success(f"🎉 You win! ({st.session_state.human_player} wins)")
        else:
            st.error(f"🤖 AI wins! ({st.session_state.ai_player} wins)")
    else:
        if st.session_state.current_player == st.session_state.human_player:
            st.info(f"🎯 Your turn ({st.session_state.human_player})")
        else:
            st.info(f"🤖 AI's turn ({st.session_state.ai_player})")
            
            # AI makes move
            if model_loaded:
                ai_move = get_ai_move(model, st.session_state.board, st.session_state.ai_player)
                if ai_move is not None:
                    st.session_state.board[ai_move] = st.session_state.ai_player
                    st.session_state.current_player = st.session_state.human_player
                    st.rerun()
            else:
                st.error("Cannot make AI move without trained model!")
    
    # Game statistics
    st.sidebar.markdown("### Game Info")
    st.sidebar.markdown(f"**Human Player:** {st.session_state.human_player}")
    st.sidebar.markdown(f"**AI Player:** {st.session_state.ai_player}")
    st.sidebar.markdown(f"**Current Turn:** {st.session_state.current_player}")
    
    if st.session_state.game_over:
        st.sidebar.markdown(f"**Game Status:** Game Over")
        if st.session_state.winner != "Draw":
            st.sidebar.markdown(f"**Winner:** {st.session_state.winner}")
        else:
            st.sidebar.markdown(f"**Result:** Draw")
    else:
        st.sidebar.markdown(f"**Game Status:** In Progress")
    
    # Instructions
    with st.expander("📋 How to Play"):
        st.markdown("""
        1. **Choose your symbol** (X or O) from the sidebar
        2. **X always goes first** - if you're X, click any empty cell to start
        3. The **AI will automatically make its move** after yours
        4. **First to get 3 in a row** (horizontally, vertically, or diagonally) wins!
        5. Click **"New Game"** to start over
        
        The AI uses a neural network trained on tic-tac-toe move data to make intelligent decisions!
        """)

if __name__ == "__main__":
    main()