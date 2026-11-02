
python -m pip install numpy gymnasium matplotlib--(run in cmd for  pkg install)

import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt

env = gym.make("CartPole-v1")

n_actions = env.action_space.n
n_states = 20
episodes = 500

learning_rate = 0.1
discount_factor = 0.99

epsilon = 1.0
epsilon_decay = 0.995
min_epsilon = 0.01



def create_bins(n_states, env):
    state_bins = []

  
    limits = [
        (-4.8, 4.8),
        (-4.0, 4.0),
        (-0.418, 0.418),
        (-4.0, 4.0)
    ]

    for low, high in limits:
        bins = np.linspace(low, high, n_states - 1)
        state_bins.append(bins)

    return state_bins


def discretize_state(state, state_bins):
    return tuple(
        np.clip(
            np.digitize(state[i], state_bins[i]),
            0,
            n_states - 1
        )
        for i in range(len(state))
    )



state_bins = create_bins(n_states, env)

q_table = np.zeros(
    (n_states, n_states, n_states, n_states, n_actions)
)

print("Q-table shape:", q_table.shape)



rewards = []

for episode in range(episodes):

    state_raw, info = env.reset()
    state = discretize_state(state_raw, state_bins)

    total_reward = 0
    done = False

    while not done:

       
        if np.random.rand() < epsilon:
            action = np.random.choice(n_actions)
        else:
            action = np.argmax(q_table[state])

      
        next_state_raw, reward, terminated, truncated, info = env.step(action)

        done = terminated or truncated

        next_state = discretize_state(next_state_raw, state_bins)

        total_reward += reward

       
        target = reward

        if not done:
            target += discount_factor * np.max(q_table[next_state])

        q_table[state][action] += learning_rate * (
            target - q_table[state][action]
        )

        state = next_state

   
    epsilon = max(min_epsilon, epsilon * epsilon_decay)

    rewards.append(total_reward)

    print(
        f"Episode {episode + 1}/{episodes}, "
        f"Reward: {total_reward}, "
        f"Epsilon: {epsilon:.3f}"
    )



plt.plot(rewards)
plt.title("Total Rewards Over Episodes")
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.grid()
plt.show()



np.save("q_table.npy", q_table)

print("Q-Table saved as 'q_table.npy'.")



test_env = gym.make("CartPole-v1", render_mode="human")

state_raw, info = test_env.reset()
state = discretize_state(state_raw, state_bins)

done = False
total_reward = 0

while not done:

    action = np.argmax(q_table[state])

    next_state_raw, reward, terminated, truncated, info = test_env.step(action)

    done = terminated or truncated

    state = discretize_state(next_state_raw, state_bins)

    total_reward += reward

test_env.close()
env.close()

print("Total reward during test:", total_reward)
