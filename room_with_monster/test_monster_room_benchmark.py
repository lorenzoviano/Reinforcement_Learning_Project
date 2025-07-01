"""
monster_room_benchmark.py – ROOM_WITH_MONSTER
  • cumulative & rolling-window curves
  • stacked-bar histogram every 1 500 episodes
  • summary table (Avg-Return + Greedy-Return)
"""
import random, copy, collections, math, numpy as np, matplotlib.pyplot as plt
from tabulate import tabulate
from minihack_env import ROOM_WITH_MONSTER, get_minihack_envirnment
from agents      import (MonteCarloOnPolicyDecay, SARSAOnPolicy, SARSAOnPolicyDecay,
                         QLearningOffPolicy,     QLearningOffPolicyDecay)

# ─── configuration ─────────────────────────────────────────────
N_EPIS   = 15_000
CHUNK    = 1_500
ROLL_W   = 50
SEEDS    = list(range(10))
GAMMA    = .99
MAX_STEP = 600
DEATH_PENALTY  = -100
GOOD_WIN_THRES = -10

AGENTS = {
    "MC-decay 0.4→0.05": (MonteCarloOnPolicyDecay,
        dict(epsilon_start=0.4, epsilon_end=0.05,
             max_episodes=N_EPIS, gamma=GAMMA), "tab:blue"),
    "SARSA-const ε0.15": (SARSAOnPolicy,
        dict(epsilon=0.15, gamma=GAMMA, alpha=0.15), "gold"),
    "SARSA-decay 0.6→0.05": (SARSAOnPolicyDecay,
        dict(epsilon_start=0.6, epsilon_end=0.05,
             max_episodes=N_EPIS, gamma=GAMMA, alpha=0.15), "tab:orange"),
    "Q-const ε0.10": (QLearningOffPolicy,
        dict(epsilon=0.10, gamma=GAMMA, alpha=0.15), "tab:green"),
    "Q-decay 0.3→0.05": (QLearningOffPolicyDecay,
        dict(epsilon_start=0.3, epsilon_end=0.05,
             max_episodes=N_EPIS, gamma=GAMMA, alpha=0.15), "lime")
}

OUTCOMES  = ["good_win", "norm_win", "fail"]        # fail = death ∪ timeout
LABEL2IDX = {lbl: i for i, lbl in enumerate(OUTCOMES)}

Metric = collections.namedtuple("Metric", "ret greedy outcomes")
def set_seed(s): random.seed(s); np.random.seed(s)
def chunk_mean(a, c): return a[-c:].mean() if len(a)>=c else np.nan

# ─── helpers for curves ────────────────────────────────────────
def cum(v): v = np.asarray(v, float); return np.cumsum(v)/np.arange(1, len(v)+1)
def mov(v, w):
    v = np.asarray(v, float)
    if len(v) < w: return np.empty(0)
    return np.convolve(v, np.ones(w)/w, mode="valid")

# ─── run one seed ──────────────────────────────────────────────
def run_one(cls, kw, seed):
    set_seed(seed)
    env = get_minihack_envirnment(ROOM_WITH_MONSTER, add_pixel=False)
    env.reset(seed=seed)
    agent = cls(id=f"{cls.__name__}-{seed}", action_space=env.action_space, **kw)

    returns, outcomes, chunk_hist = [], [], []
    for ep in range(N_EPIS):
        obs, _ = env.reset()
        G, steps, done, prev_r, died = 0.0, 0, False, 0.0, False
        ep_log = []

        while not done and steps < MAX_STEP:
            obs_cp = copy.deepcopy(obs)
            act    = agent.act(obs, prev_r)
            obs, r, term, trunc, _ = env.step(act)
            if r <= DEATH_PENALTY: died = True
            ep_log.append((obs_cp, act, r))
            G, steps, done, prev_r = G + r, steps + 1, (term or trunc), r

        if hasattr(agent, "onEpisodeEnd"):
            agent.onEpisodeEnd(episode=ep_log, reward=G,
                               episode_number=ep, last_step_reward=prev_r)
        returns.append(G)

        # ---- outcome label ------------------------------------
        if died or steps >= MAX_STEP:          # death OR stuck
            label = "fail"
        else:
            label = "good_win" if G > GOOD_WIN_THRES else "norm_win"
        outcomes.append(label)

        if (ep + 1) % CHUNK == 0:
            chunk_hist.append(outcomes[-CHUNK:])

    # ---- greedy evaluation -----------------------------------
    if hasattr(agent, "epsilon"): agent.epsilon = 0.0
    obs, _ = env.reset(); G_eval, steps, done = 0.0, 0, False
    while not done and steps < MAX_STEP:
        obs, r, term, trunc, _ = env.step(agent.act(obs, 0))
        G_eval += r; done = term or trunc; steps += 1
    env.close()
    return Metric(np.asarray(returns,float), G_eval, chunk_hist)

# ─── train all agents ──────────────────────────────────────────
records = collections.defaultdict(list)
print("Starting training loops …\n")
for name, (cls, kw, _) in AGENTS.items():
    for sd in SEEDS:
        print(f"▶ Training {name} (seed {sd}) …", end="", flush=True)
        rec = run_one(cls, kw, sd);  records[name].append(rec)
        print(f" done. avg last {CHUNK}: {chunk_mean(rec.ret, CHUNK):6.1f}")

# ─── summary table ─────────────────────────────────────────────
table = []
idx = slice(200, N_EPIS)          # ignore first 200 episodes
for name, recs in records.items():
    rets = np.stack([r.ret for r in recs])
    avg  = rets[:, idx].mean(axis=1).mean()
    sem  = rets[:, idx].mean(axis=1).std(ddof=1) / math.sqrt(len(SEEDS))
    greedy = np.array([r.greedy for r in recs])
    gmean, gsem = greedy.mean(), greedy.std(ddof=1)/math.sqrt(len(SEEDS))
    table.append([name, f"{avg:7.1f} ± {sem:.1f}", f"{gmean:7.1f} ± {gsem:.1f}"])
print(tabulate(table, headers=["Agent", "AvgReturn", "GreedyReturn"], tablefmt="github"))

# ─── learning curves ──────────────────────────────────────────
episodes = np.arange(200, N_EPIS)
plt.figure(figsize=(8,4))
for name, (_,_,col) in AGENTS.items():
    mean = np.stack([r.ret for r in records[name]]).mean(axis=0)[200:]
    plt.plot(episodes, cum(mean), lw=1.5, color=col, label=name)
plt.title("Cumulative-avg return — Monster room"); plt.legend(); plt.tight_layout()

plt.figure(figsize=(8,4))
for name, (_,_,col) in AGENTS.items():
    mean = np.stack([r.ret for r in records[name]]).mean(axis=0)[200:]
    plt.plot(episodes[ROLL_W-1:], mov(mean, ROLL_W),
             lw=1.5, color=col, label=name)
plt.title(f"{ROLL_W}-episode moving avg"); plt.legend(); plt.tight_layout()

# ─── stacked histogram ───────────────────────────────────────
bins  = np.arange(CHUNK, N_EPIS+1, CHUNK)    # 1 500, 3 000, …
bar_w = CHUNK * 0.9
colors = plt.cm.Set2(range(3))

for name in AGENTS:
    counts = np.zeros((len(bins), 3), int)
    for rec in records[name]:
        for k, lbls in enumerate(rec.outcomes):      # k = chunk index
            for lbl in lbls:
                counts[k, LABEL2IDX[lbl]] += 1
    frac = counts / (len(SEEDS) * CHUNK)

    plt.figure(figsize=(7,4))
    bottom = np.zeros(len(bins))
    for i, lbl in enumerate(OUTCOMES):
        plt.bar(bins, frac[:, i], width=bar_w, bottom=bottom,
                color=colors[i], edgecolor="k", linewidth=.4, label=lbl)
        bottom += frac[:, i]

    plt.title(f"{name} — outcome fractions per {CHUNK} episodes")
    plt.ylabel("Fraction"); plt.ylim(0, 1)
    plt.xticks(bins, bins)
    plt.legend(fontsize=8); plt.tight_layout()

plt.show()
