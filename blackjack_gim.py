import random
from collections import deque
import os
import pickle
import keyboard

# Class representing the Blackjack table/environment
class Table:
    def __init__(self):
        self.deck = deque()
        self.player_hand = []
        self.dealer_hand = []
        self.has_bet = False

    # Resets the game to a starting state
    def reset(self):
        # Creates a base list of cards (2-10, J/Q/K=11-13, Ace=14)
        base_list = [i for i in range(2, 15)] * 4
        random.shuffle(base_list)
        self.deck = deque(base_list)
        
        # Deal initial hands
        self.dealer_hand = [self.deck.popleft()] # Dealer's up card
        self.player_hand = [self.deck.popleft(), self.deck.popleft()]
        self.has_bet = False

    # Player requests a card (Hit)
    def hit_player(self):
        self.player_hand.append(self.deck.popleft())

    # Dealer requests a card
    def hit_dealer(self):
        self.dealer_hand.append(self.deck.popleft())

    # Player places a bet (Action 2 in Q-Learning)
    def place_bet(self):
        self.has_bet = True

    # Calculates all possible point totals for a given hand, handling Aces (1 or 11)
    def calculate_points(self, hand):
        totals = [0]
        for card in hand:
            if 11 <= card <= 13:  # Jack, Queen, King (value 10)
                totals = [t + 10 for t in totals]
            elif card == 14:  # Ace (value 1 or 11)
                new_totals = []
                for t in totals:
                    new_totals.append(t + 1)
                    new_totals.append(t + 11)
                totals = new_totals
            else:
                totals = [t + card for t in totals]
        
        # Return unique, sorted totals
        return sorted(set(totals))

    # Gets the best valid total for the state (highest value <= 21, or lowest if busted)
    def get_hand_value(self, hand):
        points = self.calculate_points(hand)
        valid_points = [p for p in points if p <= 21]
        
        # If there are valid points, return the highest (e.g., 18 over 8). Otherwise, return the lowest (busted).
        return max(valid_points) if valid_points else min(points)

    # Resolves the round and returns the reward
    def resolve_round(self):
        # If the dealer only has one card (due to starting state logic), deal the second card
        if len(self.dealer_hand) < 2:
            self.dealer_hand.append(self.deck.popleft())

        player_points = self.get_hand_value(self.player_hand)
        
        # Check for player bust immediately
        if player_points > 21:
            # -2 if bet was placed (loss x2), -1 if no bet (loss x1 - standard penalty)
            return -2 if self.has_bet else -1
        
        # Dealer draws until 17 or more
        while self.get_hand_value(self.dealer_hand) < 17:
            self.hit_dealer()
        
        dealer_points = self.get_hand_value(self.dealer_hand)

        # Check for dealer bust
        if dealer_points > 21:
            # 2 if bet was placed (win x2), 1 if no bet (win x1 - standard reward)
            return 2 if self.has_bet else 1
        
        # Compare points
        if player_points > dealer_points:
            return 2 if self.has_bet else 1
        elif player_points < dealer_points:
            return -2 if self.has_bet else -1
        else:
            return 0 # Push (tie)

    # Generates the state tuple for the Q-Table lookup
    def get_state(self):
        player_points_options = self.calculate_points(self.player_hand)
        player_value = self.get_hand_value(self.player_hand)
        
        # soft_ace is 1 if the player has an Ace that can count as 11 without busting, 0 otherwise.
        # This is simplified: it checks if there's more than one possible total, which implies a soft hand.
        soft_ace = len(player_points_options) - 1

        # State is: (Dealer's Up Card, Player's Hand Value, Has Soft Ace (0/1), Cards in Hand, Has Bet)
        return (self.dealer_hand[0], player_value, soft_ace, len(self.player_hand), self.has_bet)


# --- Q-LEARNING FUNCTIONS ---

# Helper to safely retrieve Q-value, defaulting to 0.0 if state-action pair is new
def get_Q(Q_table, state, action): 
    return Q_table.get((state, action), 0.0)

# Epsilon-Greedy strategy for action selection
def choose_action(Q_table, state, actions):
    global epsilon
    # Epsilon-Greedy: Choose random action with probability epsilon (Exploration)
    if random.random() < epsilon:
        return random.choice(actions)
    
    # Choose best known action (Exploitation)
    values = [get_Q(Q_table, state, a) for a in actions]
    max_value = max(values)
    
    # Handle ties by choosing randomly among the best actions
    best_actions = [a for a, v in zip(actions, values) if v == max_value]
    return random.choice(best_actions)

# The core Q-Learning update rule (Bellman Equation)
def update_Q(Q_table, state, action, reward, new_state, possible_actions):
    alpha = 0.05
    gamma = 0.9
    
    # Calculate Max Future Reward: max_a' Q(s',a')
    max_Q_new = max([get_Q(Q_table, new_state, a) for a in possible_actions])
    
    # Apply Bellman Equation: Q(s,a) += alpha * (reward + gamma * max_Q_new - Q(s,a))
    current_Q = get_Q(Q_table, state, action)
    Q_table[(state, action)] = current_Q + alpha * (reward + gamma * max_Q_new - current_Q)


# --- TRAINING PARAMETERS ---
 
epsilon = 0.7
actions = [0, 1, 2]
Q = {}



if os.path.exists("q_table.pkl"):
    print("Loading existing Q-Table...")
    with open("q_table.pkl", "rb") as f:
        Q = pickle.load(f)

training_step_count = 25833 #! You can change this number to keep record of loops
print(f"Starting at training step: {training_step_count}")

# --- MAIN TRAINING LOOP ---

while True:
    print(f"Total training steps: {training_step_count}")
    training_step_count += 1
    
    if keyboard.is_pressed('q'):
        print("Q pressed. Exiting and saving Q-Table...")
        break

    for episode in range(1000):
        # Reduce epsilon
        epsilon = max(0.05, epsilon - 0.0001)
        
        # Reset gim
        game_table = Table()
        game_table.reset()
        state = game_table.get_state()
        player_has_taken_action = False

        while True:
            action = choose_action(Q, state, actions)

            if action == 0: # Action: Hit
                game_table.hit_player()
                
                if game_table.get_hand_value(game_table.player_hand) > 21:
                    reward = -1 
                    new_state = game_table.get_state()
                    update_Q(Q, state, action, reward, new_state, actions)
                    break
                    
                reward = 0
                new_state = game_table.get_state()

            elif action == 1: # Action: Stand/Resolve
                reward = game_table.resolve_round()
                new_state = game_table.get_state()
                update_Q(Q, state, action, reward, new_state, actions)
                break # End episode

            elif action == 2 and not player_has_taken_action: # Action: Bet
                game_table.place_bet()
                reward = 0
                new_state = game_table.get_state()
            
            else: #Invalid Action
                reward = -2
                new_state = game_table.get_state()
                update_Q(Q, state, action, reward, new_state, actions)
                break

            update_Q(Q, state, action, reward, new_state, actions)
            state = new_state
            player_has_taken_action = True
        
    # Save the Q
    with open("q_table.pkl", "wb") as f:
        pickle.dump(Q, f)
        print(f"Q-Table saved at step {training_step_count}.")
