import tkinter as tk
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')


class GridWorld:
    def __init__(self, master):
        self.master = master
        self.master.title("GridWorld")

        self.grid_size = 7 # Initializing a 7x7 random walk environment
        self.cell_size = 70

        canvas_height = self.grid_size * self.cell_size + 50
        self.canvas = tk.Canvas(master, width=self.grid_size * self.cell_size, height=canvas_height)
        self.canvas.pack()

        self.rewards = np.zeros((self.grid_size, self.grid_size))
        self.transitions = {}

        # Define special states and rewards
        self.start_state = (3, 3)
        self.black_states = [(0, 6), (6, 0)]

        for state in self.black_states:
            if state[0] == 0 and state[1] == 6:
                self.rewards[state] = 1
            elif state[0] == 6 and state[1] == 0:
                self.rewards[state] = -1

        self.actions = ['U', 'D', 'L', 'R']  # Up, Down, Left, Right
        self.action_probs = {'U': 0.25, 'D': 0.25, 'L': 0.25, 'R': 0.25} # Equal probability of taking an action 

        self.values = np.zeros((self.grid_size, self.grid_size))  # Value function

        self.highest_value_label = tk.Label(self.master, text="")
        self.highest_value_label.pack()

        self.method_var = tk.StringVar(master)
        self.method_var.set("Gradient Monte Carlo")
        self.method_menu = tk.OptionMenu(master, self.method_var, "Gradient Monte Carlo", "Semi-Gradient TD(0)", "Exact Value Function") # This creates a drop-down list of all methods 
        self.method_menu.pack()

        self.start_button = tk.Button(master, text="Start", command=self.start_evaluation)
        self.start_button.pack()

        self.reset_button = tk.Button(master, text="Reset", command=self.reset_values)
        self.reset_button.pack()

        self.draw_grid()
        self.update_values()

    def draw_grid(self):
        # Define colors for special states
        self.colors = {
            (0, 6): "black",
            (6, 0): "black",
            (3, 3): "blue"
        }

        for i in range(self.grid_size):
            for j in range(self.grid_size):
                x0 = j * self.cell_size
                y0 = i * self.cell_size
                x1 = x0 + self.cell_size
                y1 = y0 + self.cell_size
                color = self.colors.get((i, j), "white")
                self.canvas.create_rectangle(x0, y0, x1, y1, fill=color, outline="black")
                self.canvas.create_text(x0 + self.cell_size / 2, y0 + self.cell_size / 2,
                                        text=f"{self.values[i, j]:.2f}", tags="values")

    def update_values(self):
        self.canvas.delete("values")
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                x0 = j * self.cell_size
                y0 = i * self.cell_size
                self.canvas.create_text(x0 + self.cell_size / 2, y0 + self.cell_size / 2,
                                        text=f"{self.values[i, j]:.2f}", tags="values")

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
            return i, j, 0  # Stay in the same position, no reward
        elif ni == 0 and nj == 6:
            return ni, nj, 1  # upper right state
        elif ni == 6 and nj == 0:
            return ni, nj, -1  # lower left state
        else:
            return ni, nj, 0  # Normal state, 0 reward

    def choose_action(self, state, epsilon=0.1):
            return random.choice(self.actions)
      
    def start_evaluation(self):
        method = self.method_var.get()
        if method == "Gradient Monte Carlo":
            self.gradient_monte_carlo()
        elif method == "Semi-Gradient TD(0)":
            self.semi_gradient_td_0()
        elif method == "Exact Value Function":
            self.exact_value_function()

    def feature_vector(self, state):
        i, j = state
        manhattan_dist_1 = abs(i - 0) + abs(j - 6)
        manhattan_dist_2 = abs(i - 6) + abs(j - 0)
        return np.array([i, j, manhattan_dist_1, manhattan_dist_2])

    def gradient_monte_carlo(self, num_episodes=10000, alpha=0.001, update_frequency=100):
        w = np.zeros((4,))  # 4 features

        for episode in range(num_episodes):
            state = self.start_state
            episode_data = []

            while state not in self.black_states:
                action = self.choose_action(state)
                next_state, reward = self.step(state, action)
                episode_data.append((state, reward))
                state = next_state

            G = 0
            for state, reward in reversed(episode_data):
                G = reward + G
                features = self.feature_vector(state)
                v_hat = np.dot(w, features)  # Approximation of value function
                grad_v_hat = features  # Gradient of v_hat with respect to w is the features vector
                w += alpha * (G - v_hat) * grad_v_hat  # Gradient update

            if episode % update_frequency == 0:
                self.values = np.dot(np.array([self.feature_vector((i, j)) for i in range(self.grid_size) for j in range(self.grid_size)]), w).reshape(self.grid_size, self.grid_size)
                self.update_values()
                self.master.update()

        self.values = np.dot(np.array([self.feature_vector((i, j)) for i in range(self.grid_size) for j in range(self.grid_size)]), w).reshape(self.grid_size, self.grid_size)
        self.update_values()
        self.display_highest_value_states()
        self.plot_heatmap()
        return

    def semi_gradient_td_0(self, num_episodes=10000, alpha=0.001, gamma=0.95, update_frequency=100):
        w = np.zeros((4,))  # 4 features, initializing the weights to zero

        for episode in range(num_episodes):
            state = self.start_state

            while state not in self.black_states:
                action = self.choose_action(state)
                next_state, reward = self.step(state, action)
                features = self.feature_vector(state)
                features_next = self.feature_vector(next_state)
                v_hat = np.dot(w, features)  # Approximation of value function
                v_hat_next = np.dot(w, features_next)  # Approximation of value function for next state
                w += alpha * (reward + gamma * v_hat_next - v_hat) * features  # Gradient ascent update
                state = next_state

            if episode % update_frequency == 0:
                self.values = np.dot(np.array([self.feature_vector((i, j)) for i in range(self.grid_size) for j in range(self.grid_size)]), w).reshape(self.grid_size, self.grid_size)
                self.update_values()
                self.master.update()

        self.values = np.dot(np.array([self.feature_vector((i, j)) for i in range(self.grid_size) for j in range(self.grid_size)]), w).reshape(self.grid_size, self.grid_size)
        self.update_values()
        self.display_highest_value_states()
        self.plot_heatmap()
        return

    def exact_value_function(self):
        P = np.zeros((self.grid_size * self.grid_size, self.grid_size * self.grid_size))
        R = np.zeros(self.grid_size * self.grid_size)

        def state_to_index(state):
            return state[0] * self.grid_size + state[1]

        for i in range(self.grid_size):
            for j in range(self.grid_size):
                state = (i, j)
                state_index = state_to_index(state)
                for action in self.actions:
                    next_state, reward = self.step(state, action)  # Use the step method, which calls the get_state_state method to get next state, action, and reward
                    next_state_index = state_to_index(next_state)
                    P[state_index, next_state_index] += 0.25  # Assuming uniform action probabilities
                    R[state_index] += 0.25 * reward

        I = np.eye(self.grid_size * self.grid_size)
        V = np.linalg.solve(I - 0.95 * P, R)  # Solving for V (Bellman Wq), V = R + gamma*P*V

        self.values = V.reshape(self.grid_size, self.grid_size)
        self.update_values()
        self.display_highest_value_states()
        self.plot_heatmap()


    def step(self, state, action):
        i, j = state
        next_i, next_j, reward = self.get_next_state(i, j, action)
        return (next_i, next_j), reward

    def reset_values(self):
        self.values = np.zeros((self.grid_size, self.grid_size))
        self.update_values()

    def display_highest_value_states(self):
        highest_value_states = np.argwhere(self.values == np.max(self.values))
        highest_value_text = f"States with the highest value: {highest_value_states.tolist()}, Value: {np.max(self.values):.2f}"
        self.highest_value_label.config(text=highest_value_text)

        print("States with the highest value:")
        for state in highest_value_states:
            print(f"State: {state}, Value: {self.values[state[0], state[1]]}")

        # print("\nFinal value function:")
        # print(self.values)

    def plot_heatmap(self):
        method = self.method_var.get()
        if method == "Gradient Monte Carlo":
            plt.imshow(self.values, cmap='Greens', interpolation='nearest')
            plt.colorbar(label='Value')
            for i in range(self.grid_size):
                for j in range(self.grid_size):
                    plt.text(j, i, f"{self.values[i, j]:.2f}", ha='center', va='center', color='black')
            plt.title('State Value Heatmap for Gradient Monte Carlo')
            plt.savefig('new_heatmap-GradientMC.png')
        elif method == "Semi-Gradient TD(0)":
            plt.imshow(self.values, cmap='Greens', interpolation='nearest')
            plt.colorbar(label='Value')
            plt.title('State Value Heatmap for TD(0)')
            for i in range(self.grid_size):
                for j in range(self.grid_size):
                    plt.text(j, i, f"{self.values[i, j]:.2f}", ha='center', va='center', color='black')
            plt.savefig('new_heatmap-GradientTD.png')
        elif method == "Exact Value Function":
            plt.imshow(self.values, cmap='Greens', interpolation='nearest')
            plt.colorbar(label='Value')
            plt.title('State Value Heatmap for Exact Value Function')
            for i in range(self.grid_size):
                for j in range(self.grid_size):
                    plt.text(j, i, f"{self.values[i, j]:.2f}", ha='center', va='center', color='black')
            plt.savefig('new_heatmap-ExactValue.png')
        plt.close()


if __name__ == "__main__":
    root = tk.Tk()
    grid_world = GridWorld(root)
    root.mainloop()
