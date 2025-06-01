import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from wrapped_flappy_bird import GameState, SCREENHEIGHT, SCREENWIDTH

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        return self.net(x)


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = map(np.array, zip(*batch))
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)


class FlappyBirdAgent:
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.policy_net = DQN(state_dim, action_dim).to(device)
        self.target_net = DQN(state_dim, action_dim).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=1e-3)
        self.replay_buffer = ReplayBuffer(10000)

        self.batch_size = 64
        self.gamma = 0.99  # discount factor
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.9995
        self.update_target_every = 1000  # steps to update target network
        self.step_count = 0

    def select_action(self, state):
        self.step_count += 1
        if random.random() < self.epsilon:
            return random.randrange(self.action_dim)
        else:
            state_v = torch.tensor(state, dtype=torch.float32).to(device).unsqueeze(0)
            with torch.no_grad():
                q_values = self.policy_net(state_v)
            return q_values.argmax().item()

    def push_experience(self, state, action, reward, next_state, done):
        self.replay_buffer.push(state, action, reward, next_state, done)

    def train_step(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        states = torch.tensor(states, dtype=torch.float32).to(device)
        actions = torch.tensor(actions, dtype=torch.int64).unsqueeze(1).to(device)
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(device)
        next_states = torch.tensor(next_states, dtype=torch.float32).to(device)
        dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1).to(device)

        # Q(s,a)
        current_q_values = self.policy_net(states).gather(1, actions)
        # max_a' Q_target(s',a')
        with torch.no_grad():
            max_next_q_values = self.target_net(next_states).max(1)[0].unsqueeze(1)
        expected_q_values = rewards + (self.gamma * max_next_q_values * (1 - dones))

        loss = nn.MSELoss()(current_q_values, expected_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

            # Epsilon decay
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        # Update target network periodically
        if self.step_count % self.update_target_every == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())


def get_state(game):
    # Создаем вектор признаков из текущего состояния игры
    # Возьмем: позицию птицы, скорость птицы, ближайшие трубы
    upper_pipe = game.upperPipes[0]
    lower_pipe = game.lowerPipes[0]

    state = np.array([
        game.playery / SCREENHEIGHT,  # нормализуем
        game.playerVelY / 10,
        upper_pipe['x'] / SCREENWIDTH,
        upper_pipe['y'] / SCREENHEIGHT,
        lower_pipe['x'] / SCREENWIDTH,
        lower_pipe['y'] / SCREENHEIGHT,
        game.playerx / SCREENWIDTH
    ], dtype=np.float32)
    return state


# === Пример игрового цикла ===

agent = FlappyBirdAgent(state_dim=7, action_dim=2)
game = GameState()

num_episodes = 1000
max_steps_per_episode = 10000

best_reward = -float('inf')  # начальное значение лучшей награды

for episode in range(num_episodes):
    game.__init__()  # сброс игры
    state = get_state(game)
    total_reward = 0

    for step in range(max_steps_per_episode):
        action_idx = agent.select_action(state)
        input_actions = [0, 0]
        input_actions[action_idx] = 1

        _, reward, done = game.frame_step(input_actions)

        if action_idx == 1:
            reward -= 0.1

        # Дополнительный штраф при столкновении — в зависимости от расстояния до центра щели 
        if done and reward == -1:
            gap_center = (game.upperPipes[0]['y'] + game.lowerPipes[0]['y']) / 2
            dy_to_gap_center = abs(game.playery - gap_center) / SCREENHEIGHT 
            reward -= dy_to_gap_center  # штраф до -2 при сильном отклонении

        total_reward += reward

        next_state = get_state(game)
        agent.push_experience(state, action_idx, reward, next_state, done)
        agent.train_step()

        state = next_state

        if done:
            # Уменьшать epsilon только если награда улучшилась
            if total_reward > best_reward:
                best_reward = total_reward
                if agent.epsilon > agent.epsilon_min:
                    agent.epsilon *= agent.epsilon_decay

            print(f"Episode {episode + 1}, Steps: {step}, Total reward: {total_reward:.2f}, "
                  f"Best reward: {best_reward:.2f}, Epsilon: {agent.epsilon:.3f}")
            break