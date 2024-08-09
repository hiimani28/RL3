import tkinter as tk
import numpy as np
import random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import mode
import seaborn as sns


class GridWorld:
    def __init__(self, master):
        self.master = master
        self.master.title("GridWorld")

        self.grid_size = 5 # Creating a 5x5 grid world, with each cell being size 50
        self.cell_size = 70

        canvas_height = self.grid_size * self.cell_size + 50
        self.canvas = tk.Canvas(master, width=self.grid_size * self.cell_size, height=canvas_height)
        self.canvas.pack()

        self.rewards = np.zeros((self.grid_size, self.grid_size))
        self.transitions = {}

       # Defining the start state, the red state (wall) and the terminal states
        self.start_state = (4, 0) 
        self.red_states = [(2, 0), (2, 1), (2, 3), (2, 4)]
        self.black_states = [(0, 0), (0, 4)]

        for state in self.red_states:
            self.rewards[state] = -20 # Setting a reward of red state being -20

        for state in self.black_states:
            self.rewards[state] = 0 # Terminal states, 0 reward

        self.actions = ['U', 'D', 'L', 'R']  # Up, Down, Left, Right
        self.action_probs = {'U': 0.25, 'D': 0.25, 'L': 0.25, 'R': 0.25}

        self.q_values = np.zeros((self.grid_size, self.grid_size, len(self.actions)))  # Initializing q values as zero
        self.policy = np.full((self.grid_size, self.grid_size), '', dtype=str)  # Policy array with empty strings

        self.highest_value_label = tk.Label(self.master, text="")
        self.highest_value_label.pack()

        self.policy_label = tk.Label(self.master, text="")
        self.policy_label.pack()

        # Dropdown menu to select evaluation method - SARSA/ Q-learning
        self.method_var = tk.StringVar(master)
        self.method_var.set("SARSA")
        self.method_menu = tk.OptionMenu(master, self.method_var, "SARSA", "Q-learning")
        self.method_menu.pack()

        self.start_button = tk.Button(master, text="Start", command=self.start_evaluation)
        self.start_button.pack()

        self.reset_button = tk.Button(master, text="Reset", command=self.reset_values)
        self.reset_button.pack()

        self.draw_grid()
        self.update_policy_display()

    def draw_grid(self):
        # Defining the colors for special states
        self.colors = {
            (0, 0): "black",
            (0, 4): "black",
            (2, 0): "red",
            (2, 1): "red",
            (2, 3): "red",
            (2, 4): "red",
            (4, 0): "blue"
        }

        self.text_ids = {}  # To store text IDs for policy update

        for i in range(self.grid_size):
            for j in range(self.grid_size):
                x0 = j * self.cell_size
                y0 = i * self.cell_size
                x1 = x0 + self.cell_size
                y1 = y0 + self.cell_size
                color = self.colors.get((i, j), "white")
                self.canvas.create_rectangle(x0, y0, x1, y1, fill=color, outline="black")
                if (i, j) not in self.black_states:
                    text_id = self.canvas.create_text(x0 + self.cell_size / 2, y0 + self.cell_size / 2, text="", font=("Helvetica", 16))
                    self.text_ids[(i, j)] = text_id

    def get_next_state(self, i, j, action):
        ni, nj = i, j
        if action == 'U':
            ni -= 1
        elif action == 'D':
            ni += 1
        elif action == 'L':
            nj -= 1
        elif action == 'R':
            nj += 1

        if ni < 0 or ni >= self.grid_size or nj < 0 or nj >= self.grid_size:
            return i, j, -1  # Stay in the same position, negative reward
        elif (ni, nj) in self.red_states:
            return self.start_state[0], self.start_state[1], -20 # Move to start state, negative reward
        elif (ni, nj) in self.black_states:
            return ni, nj, 0  # Terminal states, 0 rewards. 
        else:
            return ni, nj, -1  # Normal state, -1 reward

    def choose_action(self, state, epsilon=0.1): # If the value of action is less than epsilon, it will choose a random action!
        if random.uniform(0, 1) < epsilon:
            return random.choice(self.actions)
        else:
            state_q_values = self.q_values[state[0], state[1]]
            return self.actions[np.argmax(state_q_values)]

    def start_evaluation(self):
        method = self.method_var.get()
        if method == "SARSA":
            sarsa_rewards = self.sarsa()
            self.plot_rewards(sarsa_rewards, 'SARSA')
        elif method == "Q-learning":
            q_learning_rewards = self.q_learning()
            self.plot_rewards(q_learning_rewards, 'Q-learning')

    # Permuting the Sarsa epsilon makes it perform worse and here is the reason:
    # Why It Might Be Different from SARSA:
    # SARSA: In SARSA, increasing epsilon too much can lead to poor performance because it directly impacts the Q-value updates based on the chosen action, which might be suboptimal.
    # Q-Learning: In Q-learning, the agent updates Q-values based on the maximum future reward (max_a Q(S',a)), which is less directly affected by the current epsilon value.
    # Thus, a small increase in epsilon primarily helps in discovering better actions without as much disruption to the policy updates.
    def sarsa(self, num_episodes=10000, alpha=0.1, gamma=0.96, epsilon=0.01):
        sarsa_rewards = []
        for episode in range(num_episodes):
            state = self.start_state
            action = self.choose_action(state, epsilon)
            total_reward = 0

            while state not in self.black_states:
                next_state, reward = self.step(state, action)
                next_action = self.choose_action(next_state, epsilon)

                q_sa = self.q_values[state[0], state[1], self.actions.index(action)]  # Q(S,A)
                q_s_a = self.q_values[next_state[0], next_state[1], self.actions.index(next_action)]  # Q(S',A')

                self.q_values[state[0], state[1], self.actions.index(action)] += alpha * (reward + gamma * q_s_a - q_sa)
                # print(f"Q Value for state {state[0]},{state[1]} is: {self.q_values[state[0], state[1], self.actions.index(action)]}")

                total_reward += reward
                state = next_state
                action = next_action

            sarsa_rewards.append(total_reward)

        self.update_policy_display()
        print(sarsa_rewards)
        return sarsa_rewards

    # Permuted the epsilon Value
    # Try 0.07, 0.2 and 0.3
    def q_learning(self, num_episodes=10000, alpha=0.1, gamma=0.96, epsilon=0.01):
        q_learning_rewards = []
        for episode in range(num_episodes):
            state = self.start_state
            total_reward = 0

            while state not in self.black_states:
                action = self.choose_action(state, epsilon)
                next_state, reward = self.step(state, action)

                q_sa = self.q_values[state[0], state[1], self.actions.index(action)]  # Q(S,A)
                max_q_s_a = np.max(self.q_values[next_state[0], next_state[1]])  # max_a Q(S',a)

                self.q_values[state[0], state[1], self.actions.index(action)] += alpha * (reward + gamma * max_q_s_a - q_sa)
                # print(f"Q Value for state {state[0]},{state[1]} is: {self.q_values[state[0], state[1], self.actions.index(action)]}")

                total_reward += reward
                state = next_state

            q_learning_rewards.append(total_reward)

        self.update_policy_display()
        print(q_learning_rewards)
        return q_learning_rewards

    def step(self, state, action):
        i, j = state
        next_i, next_j, reward = self.get_next_state(i, j, action)
        return (next_i, next_j), reward

    def reset_values(self):
        self.q_values = np.zeros((self.grid_size, self.grid_size, len(self.actions)))
        self.policy.fill('')  # Clear the policy
        self.update_policy_display()

    def update_policy_display(self):
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                if (i, j) in self.black_states:
                    continue
                if np.any(self.q_values[i, j] != 0):  # Only update policy if Q-values are non-zero
                    self.policy[i, j] = self.actions[np.argmax(self.q_values[i, j])]
                else:
                    self.policy[i, j] = ''
                text_id = self.text_ids[(i, j)]        
                action = self.policy[i,j] # Just drawing the arrow, instead of displaying U,D,L,R for better viusalization
                if action == 'U':
                    arrow = '↑'
                elif action == 'D':
                    arrow = '↓'
                elif action == 'L':
                    arrow = '←'
                elif action == 'R':
                    arrow = '→'
                else:
                    arrow = ' '
                # self.canvas.itemconfig(text_id, text=self.policy[i, j])
                self.canvas.itemconfig(text_id,text = arrow)

    # def update_policy_display(self):
    #     # self.canvas.delete("policy")
    #     for i in range(self.grid_size):
    #         for j in range(self.grid_size):
    #             x0 = j * self.cell_size
    #             y0 = i * self.cell_size
    #             action = self.policy[i, j]
    #             if action == 'U':
    #                 arrow = '↑'
    #             elif action == 'D':
    #                 arrow = '↓'
    #             elif action == 'L':
    #                 arrow = '←'
    #             elif action == 'R':
    #                 arrow = '→'
    #             else:
    #                 arrow = ' '
    #             self.canvas.create_text(x0 + self.cell_size / 2, y0 + self.cell_size / 2,
    #                                     text=arrow, tags="policy", fill="black", font=("Helvetica", 24))

    def plot_rewards(self, rewards, method):
        plt.plot(rewards, label=method)
        reward_mode = mode(rewards, keepdims=True).mode[0]
        plt.axhline(reward_mode, color='green', linestyle='--', label=f'Mode: {reward_mode}')
        
        plt.xlabel('Episodes')
        plt.ylabel('Sum of Rewards')
        plt.title('Sum of Rewards over Episodes for SARSA and Q-learning')
        plt.legend()
        plt.gca().invert_yaxis()
        plt.savefig(f'{method}_rewards.png')
        plt.close()


if __name__ == "__main__":
    root = tk.Tk()
    grid_world = GridWorld(root)
    root.mainloop()
