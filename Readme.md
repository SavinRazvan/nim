### Nim Game AI Project

#### Project Overview
This project involves creating an AI that learns to play the game of Nim using reinforcement learning, specifically Q-learning. In Nim, players take turns removing objects from piles, and the player forced to take the last object loses. The AI will play multiple games against itself to learn optimal strategies.

#### Key Components
- **Nim Class**: Manages game state, available actions, and gameplay logic.
- **NimAI Class**: Implements Q-learning to train the AI, updating Q-values based on rewards from actions taken.

#### Training and Playing 
- **Training**: AI trains by playing numerous games against itself.
- **Playing**: Human players can challenge the trained AI.

#### Instructions

##### How to Run the Code
1. **Ensure you have Python installed**: This project requires Python 3.
2. **Clone the repository**: Download the project files from the repository.
3. **Navigate to the project directory**: Open your terminal or command prompt and change the directory to where the project files are located.
    ```bash
    cd path/to/your/project
    ```
4. **Run the training script**: Train the AI by running the `play.py` script. This will train the AI by playing 10,000 games against itself and then start a game where you can play against the AI.
    ```bash
    python play.py
    ```

##### How to Play the Game
1. **Start the game**: After running `play.py`, the script will train the AI and then start the game.
2. **Game Instructions**:
   - The game will display the current state of the piles.
   - When it's your turn, you will be prompted to choose a pile and the number of objects to remove from that pile.
   - Enter the pile number (0-indexed) and the number of objects to remove when prompted.
   - The AI will then make its move, and the updated state of the piles will be displayed.
   - The game continues until all objects are removed from the piles. The player forced to take the last object loses.
3. **Example of making a move**:
   - When prompted with `Choose Pile:`, enter the pile number (e.g., `1`).
   - When prompted with `Choose Count:`, enter the number of objects to remove from that pile (e.g., `3`).
   - The game will then proceed with the AI making its move.

#### Viewing Two AIs Playing Against Each Other
- You can see two AI agents play against each other by running the `nim.ipynb` notebook.
- Open the `nim.ipynb` notebook in Jupyter Notebook or Jupyter Lab.
- Follow the instructions in the notebook to train the AI agents and watch them play against each other.

The code structure and gameplay logic are based on the project specifications provided by the [CS50 AI course](https://cs50.harvard.edu/ai/2024/projects/4/nim/).

Enjoy playing the Nim game and challenging your AI!
