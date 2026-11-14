"""
Maze Escape  -  Incremental Mean (Monte Carlo Control)
=======================================================
An agent learns to escape a 6x6 maze using only experience from complete
episodes. Action values Q(s, a) are estimated with the INCREMENTAL MEAN:

        N(s,a) <- N(s,a) + 1
        Q(s,a) <- Q(s,a) + (1 / N(s,a)) * (G - Q(s,a))

Exploration: epsilon-greedy with decay.
Run:  python maze_escape_incremental_mean.py
"""

import random
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")                      # works without a display
import matplotlib.pyplot as plt
import numpy as np

# ----------------------------------------------------------------------
# 1. GAME ENVIRONMENT
# ----------------------------------------------------------------------
# Legend:  S = start   G = goal (exit)   # = wall   T = trap   . = free cell
MAZE = [
    "S . . # . .",
    ". # . # T .",
    ". # . . . .",
    ". . T # # .",
    "# . . . . .",
    ". . # T . G",
]
MAZE = [row.split() for row in MAZE]
ROWS, COLS = len(MAZE), len(MAZE[0])

# ----------------------------------------------------------------------
# 2. ACTION SPACE
# ----------------------------------------------------------------------
ACTIONS = {0: ("UP", (-1, 0)), 1: ("DOWN", (1, 0)),
           2: ("LEFT", (0, -1)), 3: ("RIGHT", (0, 1))}
ARROWS = {0: "↑", 1: "↓", 2: "←", 3: "→"}
N_ACTIONS = len(ACTIONS)

# ----------------------------------------------------------------------
# 3. REWARD FUNCTION
# ----------------------------------------------------------------------
REWARD_STEP = -1      # every move costs 1  -> encourages short paths
REWARD_WALL = -5      # bumping into a wall / the border (agent stays put)
REWARD_TRAP = -20     # stepping on a trap (episode ends)
REWARD_GOAL = +50     # reaching the exit  (episode ends)
MAX_STEPS = 60        # safety limit per episode


class MazeEscape:
    """Maze game: states = (row, col), actions = UP/DOWN/LEFT/RIGHT."""

    def __init__(self, layout):
        self.layout = layout
        self.start = self._find("S")
        self.goal = self._find("G")
        self.state = self.start
        self.steps = 0

    def _find(self, symbol):
        for r in range(ROWS):
            for c in range(COLS):
                if self.layout[r][c] == symbol:
                    return (r, c)

    def reset(self):
        self.state, self.steps = self.start, 0
        return self.state

    # ---- transition rules -------------------------------------------
    def step(self, action):
        self.steps += 1
        dr, dc = ACTIONS[action][1]
        r, c = self.state
        nr, nc = r + dr, c + dc

        # Rule 1: border or wall -> agent stays, gets wall penalty
        if not (0 <= nr < ROWS and 0 <= nc < COLS) or self.layout[nr][nc] == "#":
            reward, done = REWARD_WALL, False
        else:
            self.state = (nr, nc)
            cell = self.layout[nr][nc]
            if cell == "G":                       # Rule 2: exit found
                reward, done = REWARD_GOAL, True
            elif cell == "T":                     # Rule 3: trap
                reward, done = REWARD_TRAP, True
            else:                                 # Rule 4: normal move
                reward, done = REWARD_STEP, False

        if self.steps >= MAX_STEPS:               # Rule 5: time-out
            done = True
        return self.state, reward, done


# ----------------------------------------------------------------------
# 4. AGENT: Q-values, counters, epsilon-greedy
# ----------------------------------------------------------------------
Q = defaultdict(lambda: np.zeros(N_ACTIONS))   # Q-values initialised to 0
N = defaultdict(lambda: np.zeros(N_ACTIONS))   # visit counters N(s,a) = 0


def epsilon_greedy(state, epsilon):
    if random.random() < epsilon:                          # explore
        return random.randrange(N_ACTIONS)
    q = Q[state]                                           # exploit (random tie-break)
    best = np.flatnonzero(q == q.max())
    return int(random.choice(best))


def generate_episode(env, epsilon):
    """Play one full game; return list of (state, action, reward)."""
    episode, state, done = [], env.reset(), False
    while not done:
        action = epsilon_greedy(state, epsilon)
        next_state, reward, done = env.step(action)
        episode.append((state, action, reward))
        state = next_state
    return episode


# ----------------------------------------------------------------------
# 5. TRAINING  (First-Visit Monte Carlo + Incremental Mean)
# ----------------------------------------------------------------------
def train(episodes=1000, gamma=0.99, eps_start=1.0, eps_min=0.05, eps_decay=0.995):
    env = MazeEscape(MAZE)
    rewards_per_episode, success = [], []
    epsilon = eps_start

    for ep in range(episodes):
        episode = generate_episode(env, epsilon)

        # ---- returns, computed backwards: G_t = r_t + gamma * G_{t+1}
        G, returns = 0.0, []
        for (_, _, r) in reversed(episode):
            G = r + gamma * G
            returns.append(G)
        returns.reverse()

        # ---- first-visit update of every eligible (s, a) pair
        visited = set()
        for t, (s, a, _) in enumerate(episode):
            if (s, a) in visited:
                continue
            visited.add((s, a))
            N[s][a] += 1                                   # counter
            Q[s][a] += (returns[t] - Q[s][a]) / N[s][a]    # INCREMENTAL MEAN

        rewards_per_episode.append(sum(r for (_, _, r) in episode))
        success.append(1 if env.state == env.goal else 0)
        epsilon = max(eps_min, epsilon * eps_decay)        # decay exploration

        if (ep + 1) % 100 == 0:
            avg = np.mean(rewards_per_episode[-100:])
            sr = 100 * np.mean(success[-100:])
            print(f"Episode {ep+1:4d} | avg reward (last 100): {avg:8.2f} "
                  f"| success: {sr:5.1f}% | epsilon: {epsilon:.3f}")
    return rewards_per_episode, success


# ----------------------------------------------------------------------
# 6. DISPLAY RESULTS
# ----------------------------------------------------------------------
def print_q_values():
    print("\n" + "=" * 62)
    print("FINAL Q-VALUES   (state = (row, col))")
    print("=" * 62)
    print(f"{'State':<10}{'UP':>10}{'DOWN':>10}{'LEFT':>10}{'RIGHT':>10}   Best")
    for r in range(ROWS):
        for c in range(COLS):
            if MAZE[r][c] in ("#", "G", "T"):
                continue
            q = Q[(r, c)]
            print(f"{str((r, c)):<10}" + "".join(f"{v:10.2f}" for v in q)
                  + f"   {ACTIONS[int(np.argmax(q))][0]}")


def print_policy():
    print("\n" + "=" * 62)
    print("LEARNED POLICY (greedy w.r.t. Q)")
    print("=" * 62)
    for r in range(ROWS):
        line = ""
        for c in range(COLS):
            cell = MAZE[r][c]
            if cell in ("#", "G", "T", "S") and cell != "S":
                line += f" {cell} "
            else:
                q = Q[(r, c)]
                line += f" {ARROWS[int(np.argmax(q))] if q.any() else '?'} "
        print(line)
    print("(# wall, T trap, G goal)")


def play_greedy():
    env = MazeEscape(MAZE)
    s, done, path, total = env.reset(), False, [env.start], 0
    while not done:
        s, r, done = env.step(int(np.argmax(Q[s])))
        path.append(s); total += r
    print("\nGreedy path :", " -> ".join(map(str, path)))
    print("Steps / total reward:", len(path) - 1, "/", total)
    return path


def plot_results(rewards, success, path):
    w = 25
    ma = np.convolve(rewards, np.ones(w) / w, mode="valid")
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))

    ax[0].plot(rewards, alpha=0.3, color="#6C63FF", label="Reward per episode")
    ax[0].plot(range(w - 1, len(rewards)), ma, color="#FF6584", lw=2,
               label=f"Moving average ({w})")
    ax[0].set_title("Cumulative Reward per Episode")
    ax[0].set_xlabel("Episode"); ax[0].set_ylabel("Cumulative reward")
    ax[0].legend(); ax[0].grid(alpha=0.3)

    sr = np.convolve(success, np.ones(50) / 50, mode="valid") * 100
    ax[1].plot(sr, color="#2EC4B6", lw=2)
    ax[1].set_title("Success Rate (moving avg of 50 episodes)")
    ax[1].set_xlabel("Episode"); ax[1].set_ylabel("Escapes (%)")
    ax[1].grid(alpha=0.3)
    plt.tight_layout(); plt.savefig("reward_plot.png", dpi=150)

    # maze + learned path figure
    fig, a = plt.subplots(figsize=(6, 6))
    colors = {".": "#F5F5F5", "#": "#2B2D42", "S": "#8ECAE6", "G": "#80ED99", "T": "#EF476F"}
    for r in range(ROWS):
        for c in range(COLS):
            a.add_patch(plt.Rectangle((c, ROWS - 1 - r), 1, 1,
                        facecolor=colors[MAZE[r][c]], edgecolor="white"))
            if MAZE[r][c] in "SGT":
                a.text(c + .5, ROWS - 1 - r + .5, MAZE[r][c], ha="center",
                       va="center", fontsize=16, weight="bold")
    xs = [c + .5 for (_, c) in path]; ys = [ROWS - 1 - r + .5 for (r, _) in path]
    a.plot(xs, ys, color="#FF6584", lw=3, marker="o")
    a.set_xlim(0, COLS); a.set_ylim(0, ROWS); a.set_aspect("equal"); a.axis("off")
    a.set_title("Learned Escape Route")
    plt.savefig("maze_path.png", dpi=150, bbox_inches="tight")


# ----------------------------------------------------------------------
if __name__ == "__main__":
    random.seed(42); np.random.seed(42)
    rewards, success = train(episodes=1000)
    print_q_values()
    print_policy()
    route = play_greedy()
    plot_results(rewards, success, route)
    print(f"\nAverage reward first 100 episodes: {np.mean(rewards[:100]):.2f}")
    print(f"Average reward last 100 episodes : {np.mean(rewards[-100:]):.2f}")
    print("Saved: reward_plot.png , maze_path.png")
