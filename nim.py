import math
import random
import time


class Nim:
    def __init__(self, initial=[1, 3, 5, 7]):
        """
        Initialize game board.
        Each game board has:
            - `piles`: a list of how many elements remain in each pile
            - `player`: 0 or 1 to indicate which player's turn
            - `winner`: None, 0, or 1 to indicate who the winner is
        """
        self.piles = initial.copy()
        self.player = 0
        self.winner = None


    @classmethod
    def available_actions(cls, piles):
        """
        Nim.available_actions(piles) takes a `piles` list as input
        and returns all of the available actions `(i, j)` in that state.

        Action `(i, j)` represents the action of removing `j` items
        from pile `i` (where piles are 0-indexed).
        """
        actions = set()
        for i, pile in enumerate(piles):
            for j in range(1, pile + 1):
                actions.add((i, j))
        return actions


    @classmethod
    def other_player(cls, player):
        """
        Nim.other_player(player) returns the player that is not
        `player`. Assumes `player` is either 0 or 1.
        """
        return 0 if player == 1 else 1


    def switch_player(self):
        """
        Switch the current player to the other player.
        """
        self.player = Nim.other_player(self.player)


    def move(self, action):
        """
        Make the move `action` for the current player.
        `action` must be a tuple `(i, j)`.
        """
        pile, count = action

        # Check for errors
        if self.winner is not None:
            raise Exception("Game already won")
        elif pile < 0 or pile >= len(self.piles):
            raise Exception("Invalid pile")
        elif count < 1 or count > self.piles[pile]:
            raise Exception("Invalid number of objects")

        # Update pile
        self.piles[pile] -= count
        self.switch_player()

        # Check for a winner
        if all(pile == 0 for pile in self.piles):
            self.winner = self.player


class NimAI:
    def __init__(self, alpha=0.5, epsilon=0.1):
        """
        Initialize AI with an empty Q-learning dictionary,
        an alpha (learning) rate, and an epsilon rate.

        The Q-learning dictionary maps `(state, action)`
        pairs to a Q-value (a number).
         - `state` is a tuple of remaining piles, e.g., (1, 1, 4, 4)
         - `action` is a tuple `(i, j)` for an action
        """
        self.q = dict()
        self.alpha = alpha
        self.epsilon = epsilon


    def update(self, old_state, action, new_state, reward):
        """
        Update Q-learning model, given an old state, an action taken
        in that state, a new resulting state, and the reward received
        from taking that action.
        """
        old = self.get_q_value(old_state, action)
        best_future = self.best_future_reward(new_state)
        self.update_q_value(old_state, action, old, reward, best_future)


    def get_q_value(self, state, action):
        """
        Return the Q-value for the state `state` and the action `action`.
        If no Q-value exists yet in `self.q`, return 0.
        """
        # Get the Q-value from the Q-learning dictionary using the tuple of state and actions as the key
        q_value = self.q.get((tuple(state), action))

        # If a Q-value exists, return it
        if q_value:
            return q_value

        # If no Q-value exists, return 0 as the default value
        else:
            return 0


    def update_q_value(self, state, action, old_q, reward, future_rewards):
        """
        Update the Q-value for the state `state` and the action `action`
        given the previous Q-value `old_q`, a current reward `reward`,
        and an estimate of future rewards `future_rewards`.

        Use the formula:

        Q(s, a) <- old value estimate
                   + alpha * (new value estimate - old value estimate)

        where `old value estimate` is the previous Q-value,
        `alpha` is the learning rate, and `new value estimate`
        is the sum of the current reward and estimated future rewards.
        """

        # Calculate the new Q-value using the Q-learning formula
        new_q = old_q + self.alpha * ((reward + future_rewards) - old_q)
        # print(f"new_q = {new_q}")

        # Update the Q-value in the Q-learning dictionary with the new value
        self.q[(tuple(state), action)] = new_q


    def best_future_reward(self, state):
        """
        Given a state `state`, consider all possible `(state, action)`
        pairs available in that state and return the maximum of all
        of their Q-values.

        Use 0 as the Q-value if a `(state, action)` pair has no
        Q-value in `self.q`. If there are no available actions in
        `state`, return 0.
        """

        # Get all available actions for the given state
        actions = Nim.available_actions(state)
        # Initialize the best action's Q-value as None
        best_action_q_value = None

        # If no actions are available, return 0
        if not actions:
            return 0

        # Iterate over each action and find the maximum Q-value
        for action in actions:
            # Get the Q-value for the current action, or 0 if it doesn't exist in self.q
            q_value = self.q.get((tuple(state), action))
            q_value = q_value if q_value else 0

            # Update the best action's Q-value if it is None or greater than the current best
            if best_action_q_value is None or q_value > best_action_q_value:
                best_action_q_value = q_value

        # Return the maximum Q-value among all available actions
        return best_action_q_value


    def choose_action(self, state, epsilon=True):
        """
        Given a state `state`, return an action `(i, j)` to take.

        If `epsilon` is `False`, then return the best action
        available in the state (the one with the highest Q-value,
        using 0 for pairs that have no Q-values).

        If `epsilon` is `True`, then with probability
        `self.epsilon` choose a random available action,
        otherwise choose the best action available.

        If multiple actions have the same Q-value, any of those
        options is an acceptable return value.
        """

        # Get all available actions for the given state
        actions = Nim.available_actions(state)

        # print(f"Available actions for this state are: {actions}")
        # Available actions for this state are: {(0, 1), (2, 4), (1, 2), (2, 1), (3, 4), (3, 1), (3, 7), (1, 1), (2, 3), (3, 3), (3, 6), (2, 2), (3, 2), (2, 5), (1, 3), (3, 5)}

        # Return a random action with probability self.epsilon when epsilon is True
        epsilon_probability_random_move = random.random()
        # print(epsilon_probability_random_move)
        if epsilon and epsilon_probability_random_move < self.epsilon:
            return random.choice(list(actions))

        # Initialize the best action and its Q-value as None
        best_action = None
        best_action_q_value = None

        # Iterate over each action and find the best action with the highest Q-value
        for action in actions:
            # Get the Q-value for the current action, or 0 if it doesn't exist in self.q
            q_value = self.q.get((tuple(state), action))
            q_value = q_value if q_value else 0

            # Update the best action and its Q-value if it is None or has a higher Q-value
            if best_action_q_value is None or q_value > best_action_q_value:
                best_action_q_value = q_value
                best_action = action

        # Return the best action available or a random action when epsilon is True
        return best_action


def train(n):
    """
    Train an AI by playing `n` games against itself.
    """

    player = NimAI()

    # Play n games
    for i in range(n):
        print(f"Playing training game {i + 1}")
        game = Nim()

        # Keep track of last move made by either player
        last = {0: {"state": None, "action": None}, 1: {"state": None, "action": None}}

        # Game loop
        while True:
            # Keep track of current state and action
            state = game.piles.copy()
            action = player.choose_action(game.piles)

            # Keep track of last state and action
            last[game.player]["state"] = state
            last[game.player]["action"] = action

            # Make move
            game.move(action)
            new_state = game.piles.copy()

            # When game is over, update Q values with rewards
            if game.winner is not None:
                player.update(state, action, new_state, -1)
                player.update(
                    last[game.player]["state"],
                    last[game.player]["action"],
                    new_state,
                    1,
                )
                break

            # If game is continuing, no rewards yet
            elif last[game.player]["state"] is not None:
                player.update(
                    last[game.player]["state"],
                    last[game.player]["action"],
                    new_state,
                    0,
                )

    print("Done training")

    # Return the trained AI
    return player


def play(ai, human_player=None):
    """
    Play a human game against the AI.
    `human_player` can be set to 0 or 1 to specify whether
    the human player moves first or second.
    """

    # If no player order is set, choose the human's order randomly
    if human_player is None:
        human_player = random.randint(0, 1)

    # Create a new game
    game = Nim()

    # Game loop
    while True:
        # Print contents of piles
        print()
        print("Piles:")
        for i, pile in enumerate(game.piles):
            print(f"Pile {i}: {pile}")
        print()

        # Compute available actions
        available_actions = Nim.available_actions(game.piles)
        time.sleep(1)

        # Let the human make a move
        if game.player == human_player:
            print("Your Turn")
            while True:
                pile = int(input("Choose Pile: "))
                count = int(input("Choose Count: "))
                if (pile, count) in available_actions:
                    break
                print("Invalid move, try again.")

        # Have the AI make a move
        else:
            print("AI's Turn")
            pile, count = ai.choose_action(game.piles, epsilon=False)
            print(f"AI chose to take {count} from pile {pile}.")

        # Make move
        game.move((pile, count))

        # Check for winner
        if game.winner is not None:
            print()
            print("GAME OVER")
            winner = "Human" if game.winner == human_player else "AI"
            print(f"Winner is {winner}")
            return
