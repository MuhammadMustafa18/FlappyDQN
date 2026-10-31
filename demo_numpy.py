import os
import json
import numpy as np
from game import FlappyBirdEnv

class PureNumpyAgent:
    """Runs forward inference using only NumPy weights (no PyTorch needed locally)."""
    def __init__(self, weights_path="model_weights.json"):
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"Weights file not found: {weights_path}")
        
        with open(weights_path, "r") as f:
            weights_data = json.load(f)

        self.w0 = np.array(weights_data["net.0.weight"], dtype=np.float32)
        self.b0 = np.array(weights_data["net.0.bias"], dtype=np.float32)
        self.w2 = np.array(weights_data["net.2.weight"], dtype=np.float32)
        self.b2 = np.array(weights_data["net.2.bias"], dtype=np.float32)
        self.w4 = np.array(weights_data["net.4.weight"], dtype=np.float32)
        self.b4 = np.array(weights_data["net.4.bias"], dtype=np.float32)
        self.w6 = np.array(weights_data["net.6.weight"], dtype=np.float32)
        self.b6 = np.array(weights_data["net.6.bias"], dtype=np.float32)

    def select_action(self, state):
        # Layer 1
        x = np.maximum(0, np.dot(self.w0, state) + self.b0)
        # Layer 2
        x = np.maximum(0, np.dot(self.w2, x) + self.b2)
        # Layer 3
        x = np.maximum(0, np.dot(self.w4, x) + self.b4)
        # Output Q-values [Q(idle), Q(flap)]
        q_vals = np.dot(self.w6, x) + self.b6
        return int(np.argmax(q_vals))

def run_demo(weights_path="model_weights.json"):
    env = FlappyBirdEnv(render_mode=True)
    agent = None
    if os.path.exists(weights_path):
        agent = PureNumpyAgent(weights_path)
        print(f"[*] Loaded trained model weights from {weights_path}!")
    else:
        print("[!] No model weights found yet. Bird will fall or flap randomly.")

    for ep in range(1, 10):
        state = env.reset()
        done = False
        print(f"--- Episode {ep} Started ---")

        while not done:
            if agent:
                action = agent.select_action(state)
            else:
                action = 0
            state, reward, done, info = env.step(action)

        print(f"Episode {ep} Finished | Pipes: {info['score']}")

    env.close()

if __name__ == "__main__":
    import sys
    w = sys.argv[1] if len(sys.argv) > 1 else "model_weights.json"
    run_demo(w)
