import time
import os
from game import FlappyBirdEnv
from dqn_agent import DQNAgent

def run_demo(checkpoint_path="model_best.pth", num_episodes=5):
    env = FlappyBirdEnv(render_mode=True)
    state = env.reset()
    state_dim = len(state)

    agent = DQNAgent(state_dim=state_dim)

    if os.path.exists(checkpoint_path):
        print(f"[*] Loading model from {checkpoint_path}...")
        agent.load(checkpoint_path)
    else:
        print("[!] No trained checkpoint found. Running with random / untrained agent...")

    for ep in range(1, num_episodes + 1):
        state = env.reset()
        done = False
        print(f"--- Playing Episode {ep} ---")

        while not done:
            # evaluate=True turns off random epsilon exploration
            action = agent.select_action(state, evaluate=True)
            next_state, reward, done, info = env.step(action)
            state = next_state

        print(f"Episode {ep} Finished! Pipes Cleared: {info['score']}")
        time.sleep(1.0)

    env.close()

if __name__ == "__main__":
    import sys
    checkpoint = sys.argv[1] if len(sys.argv) > 1 else "model_best.pth"
    run_demo(checkpoint)
