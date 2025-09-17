# abl_carracing.py
"""
Simplified Abductive Learning loop applied to CarRacing.
- Requires: gym or gymnasium with Box2D; torch; numpy; pillow; matplotlib; tqdm
- Run: python abl_carracing.py
"""
import math, time, random, json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange
import argparse

# Attempt to import gymnasium then gym
try:
    import gymnasium as gym
except Exception:
    import gym

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image

# ---------------------------
# Utilities & heuristics
# ---------------------------
def preprocess_frame(frame, size=(96,96)):
    img = Image.fromarray(frame)
    img = img.resize(size)
    arr = np.array(img).astype(np.float32) / 255.0
    arr = (arr - 0.5) * 2.0
    arr = np.transpose(arr, (2,0,1)).astype(np.float32)
    return arr

def heuristic_labels_from_frame(frame):
    img = np.array(frame).astype(np.float32) / 255.0
    R = img[:,:,0]; G = img[:,:,1]; B = img[:,:,2]
    green_mask = (G > 0.45) & (G > R + 0.05)
    green_frac = green_mask.mean()
    on_track = 0 if green_frac > 0.2 else 1

    gray = img.mean(axis=2)
    road_mask = gray < 0.8
    coords = np.column_stack(np.nonzero(road_mask))
    if len(coords) < 10:
        dir = 0
    else:
        y_coords = coords[:,0]; x_coords = coords[:,1]
        lower = y_coords > (img.shape[0]*0.5)
        if lower.sum() < 5:
            dir = 0
        else:
            x_center = x_coords[lower].mean()
            img_center = img.shape[1] / 2.0
            dx = x_center - img_center
            if dx < -5:
                dir = -1
            elif dx > 5:
                dir = 1
            else:
                dir = 0

    speed_proxy = gray.mean()
    if speed_proxy > 0.7:
        speed_bin = 0
    elif speed_proxy > 0.5:
        speed_bin = 1
    else:
        speed_bin = 2
    return int(on_track), int(dir), int(speed_bin)

# ---------------------------
# Perception net
# ---------------------------
class PerceptionNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, 5, stride=2, padding=2), nn.ReLU(),
            nn.Conv2d(16, 32, 5, stride=2, padding=2), nn.ReLU(),
            nn.Conv2d(32, 64, 5, stride=2, padding=2), nn.ReLU(),
            nn.Flatten()
        )
        with torch.no_grad():
            dummy = torch.zeros(1,3,96,96)
            flat = self.conv(dummy).shape[1]
        self.fc = nn.Sequential(nn.Linear(flat, 128), nn.ReLU())
        self.on_track_head = nn.Linear(128, 2)
        self.dir_head = nn.Linear(128, 3)
        self.speed_head = nn.Linear(128, 3)
    def forward(self, x):
        z = self.conv(x)
        z = self.fc(z)
        return self.on_track_head(z), self.dir_head(z), self.speed_head(z)

class FrameDataset(Dataset):
    def __init__(self, frames, labels):
        self.frames = frames
        self.labels = labels
    def __len__(self): return len(self.frames)
    def __getitem__(self, idx):
        x = self.frames[idx]
        y = self.labels[idx]
        x = torch.tensor(x, dtype=torch.float32)
        y_on = torch.tensor(y[0], dtype=torch.long)
        y_dir = torch.tensor(y[1]+1, dtype=torch.long)
        y_speed = torch.tensor(y[2], dtype=torch.long)
        return x, (y_on, y_dir, y_speed)

def train_perception(net, dataset, epochs=3, batch_size=64, lr=1e-3, device='cpu'):
    if len(dataset)==0: return
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    net.to(device); net.train()
    opt = optim.Adam(net.parameters(), lr=lr)
    ce = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        for xb, yb in loader:
            xb = xb.to(device)
            y_on, y_dir, y_speed = yb
            y_on = y_on.to(device); y_dir = y_dir.to(device); y_speed = y_speed.to(device)
            on_logits, dir_logits, speed_logits = net(xb)
            loss = ce(on_logits, y_on) + ce(dir_logits, y_dir) + ce(speed_logits, y_speed)
            opt.zero_grad(); loss.backward(); opt.step()
    net.cpu(); net.eval()

def evaluate_perception(net, dataset, device='cpu'):
    if len(dataset)==0: return {"on_acc":None,"dir_acc":None,"speed_acc":None}
    loader = DataLoader(dataset, batch_size=128, shuffle=False)
    net.to(device); net.eval()
    correct_on = correct_dir = correct_speed = total = 0
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            y_on, y_dir, y_speed = yb
            y_on = y_on.to(device); y_dir = y_dir.to(device); y_speed = y_speed.to(device)
            on_logits, dir_logits, speed_logits = net(xb)
            pred_on = on_logits.argmax(dim=1)
            pred_dir = dir_logits.argmax(dim=1)
            pred_speed = speed_logits.argmax(dim=1)
            correct_on += (pred_on==y_on).sum().item()
            correct_dir += (pred_dir==y_dir).sum().item()
            correct_speed += (pred_speed==y_speed).sum().item()
            total += xb.size(0)
    net.cpu()
    return {"on_acc": correct_on/total, "dir_acc": correct_dir/total, "speed_acc": correct_speed/total}

# ---------------------------
# Symbolic controller
# ---------------------------
def symbolic_controller(symbols):
    on_track, dir, speed = symbols
    steer = -0.8 if dir==-1 else (0.8 if dir==1 else 0.0)
    if on_track==0:
        gas = 0.2; brake = 0.1
    else:
        if speed==2:
            gas = 0.0; brake = 0.0
        elif speed==1:
            gas = 0.3; brake = 0.0
        else:
            gas = 0.6; brake = 0.0
    return np.array([steer, gas, brake], dtype=np.float32)

# ---------------------------
# Main experiment
# ---------------------------
def run_experiment():
    env = gym.make("CarRacing-v3", continuous=True, render_mode="rgb_array")
    perception = PerceptionNet()
    frames, labels = [], []

    # Collect small random dataset
    for ep in range(4):
        obs = env.reset()
        done=False; steps=0
        while True:
            a = env.action_space.sample()
            step = env.step(a)
            if len(step)==5: obs, reward, terminated, truncated, info = step; done = terminated or truncated
            else: obs, reward, done, info = step
            frame = env.render()
            frames.append(preprocess_frame(frame))
            labels.append(heuristic_labels_from_frame(frame))
            steps += 1
            if done or steps>800: break

    dataset = FrameDataset(frames, labels)
    train_perception(perception, dataset, epochs=6, batch_size=128, lr=1e-3)
    print("Initial eval:", evaluate_perception(perception, dataset))

    results = {"episode_rewards": [], "perception_on_acc": [], "perception_dir_acc": [], "perception_speed_acc": [], "num_corrections": []}

    for it in range(10):
        ep_rewards = []
        new_frames, new_labels = [], []
        corrections=0
        for ep in range(3):
            obs = env.reset()
            done=False; steps=0; total_reward=0.0; traj=[]
            while True:
                frame = env.render()
                pf = preprocess_frame(frame)
                x = torch.tensor(pf[None], dtype=torch.float32)
                with torch.no_grad():
                    on_logits, dir_logits, speed_logits = perception(x)
                    on_pred = int(on_logits.argmax(dim=1).item())
                    dir_pred = int(dir_logits.argmax(dim=1).item()) - 1
                    speed_pred = int(speed_logits.argmax(dim=1).item())
                symbols = (on_pred, dir_pred, speed_pred)
                a = symbolic_controller(symbols)
                step = env.step(a)
                if len(step)==5: obs, reward, terminated, truncated, info = step; done = terminated or truncated
                else: obs, reward, done, info = step
                total_reward += reward
                traj.append((pf, symbols, heuristic_labels_from_frame(frame), reward))
                steps += 1
                if done or steps>1000: break
            ep_rewards.append(total_reward)
            if total_reward < 50.0:
                for pf, symbols, hlab, r in traj:
                    if symbols != hlab:
                        corrections += 1
                        new_frames.append(pf); new_labels.append(hlab)
            else:
                for pf, symbols, hlab, r in traj:
                    new_frames.append(pf); new_labels.append(symbols)
        if new_frames:
            frames.extend(new_frames); labels.extend(new_labels)
            dataset = FrameDataset(frames, labels)
            train_perception(perception, dataset, epochs=3, batch_size=128, lr=5e-4)
        metrics = evaluate_perception(perception, dataset)
        results["episode_rewards"].extend(ep_rewards)
        results["perception_on_acc"].append(metrics["on_acc"]); results["perception_dir_acc"].append(metrics["dir_acc"]); results["perception_speed_acc"].append(metrics["speed_acc"])
        results["num_corrections"].append(corrections)
        print(f"Iter {it+1}: avg reward {np.mean(ep_rewards):.2f}, corrections {corrections}, metrics {metrics}")

    # Plots
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(3,1, figsize=(8,12))
    axs[0].plot(results["episode_rewards"]); axs[0].set_title("Episode rewards")
    axs[1].plot(results["perception_on_acc"], label="on"); axs[1].plot(results["perception_dir_acc"], label="dir"); axs[1].plot(results["perception_speed_acc"], label="speed"); axs[1].legend()
    axs[2].bar(range(1,len(results["num_corrections"])+1), results["num_corrections"]); axs[2].set_title("Corrections per iteration")
    plt.tight_layout(); plt.show()

    # Save
    out = Path("abl_carracing_run"); out.mkdir(exist_ok=True)
    torch.save(perception.state_dict(), out/"perception_final.pt")
    with open(out/"results_summary.json","w") as f:
        json.dump(results, f)
    print("Saved results to", out)

if __name__ == "__main__":
    run_experiment()
