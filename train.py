import os
import time
from collections import deque
import numpy as np
import matplotlib.pyplot as plt

from game import FlappyBirdEnv
from dqn_agent import DQNAgent

# Hyperparameters
NUM_EPISODES = 10000        # Can run up to 20,000
LOG_INTERVAL = 100
SAVE_INTERVAL = 500
CHECKPOINT_PATH = "model_best.pth"

def train():
    env = FlappyBirdEnv(render_mode=False)  # Headless mode: thousands of FPS
    state = env.reset()
    state_dim = len(state)

    agent = DQNAgent(state_dim=state_dim)
    print(f"[*] Training on device: {agent.device}")
    print(f"[*] State dimension: {state_dim} (15 LIDAR rays + 4 spatial kinematics)")
    print(f"[*] Target episodes: {NUM_EPISODES}")
    print("-" * 55)

    recent_scores = deque(maxlen=100)
    best_score = 0
    all_scores = []
    start_time = time.time()

    for episode in range(1, NUM_EPISODES + 1):
        state = env.reset()
        episode_reward = 0.0
        done = False

        while not done:
            action = agent.select_action(state)
            next_state, reward, done, info = env.step(action)
            agent.replay_buffer.push(state, action, reward, next_state, done)
            agent.train_step()

            state = next_state
            episode_reward += reward

        score = info["score"]
        recent_scores.append(score)
        all_scores.append(score)

        if score > best_score:
            best_score = score
            agent.save(CHECKPOINT_PATH)

        if episode % LOG_INTERVAL == 0:
            avg_score = np.mean(recent_scores)
            elapsed = time.time() - start_time
            print(
                f"Ep: {episode:5d} | "
                f"Avg Score (last 100): {avg_score:5.2f} | "
                f"Max Score: {best_score:3d} | "
                f"Epsilon: {agent.epsilon:5.3f} | "
                f"Buffer: {len(agent.replay_buffer):5d} | "
                f"Time: {elapsed:5.1f}s"
            )

        if episode % SAVE_INTERVAL == 0:
            agent.save(f"model_ep{episode}.pth")

    print("\n[+] Training complete!")
    print(f"[+] Best score achieved: {best_score}")
    print(f"[+] Best model saved at: {CHECKPOINT_PATH}")

    # Plot and save learning curve
    plt.figure(figsize=(10, 5))
    plt.plot(all_scores, label="Episode Score", alpha=0.3, color="blue")
    running_avg = np.convolve(all_scores, np.ones(100)/100, mode="valid")
    plt.plot(running_avg, label="100-Ep Moving Avg", color="red")
    plt.title("Flappy Bird DQN Training Progress")
    plt.xlabel("Episode")
    plt.ylabel("Score (Pipes Cleared)")
    plt.legend()
    plt.grid(True)
    plt.savefig("training_curve.png")
    print("[+] Saved training plot to training_curve.png")

if __name__ == "__main__":
    train()
