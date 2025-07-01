# dyna_vs_q_benchmark.py
# ==============================================================
"""
Benchmark: Dyna-Q   vs   Q-learning
  • Room-with-Cliff   (2 000 ep)
  • Room-with-Monster (15 000 ep)
  • 10 seeds averaged
  • 50-episode moving-average curves
  • outcome histogram for Monster
"""
import random, copy, collections, math, numpy as np, matplotlib.pyplot as plt
from tabulate import tabulate

# ─── your own code base ────────────────────────────────────────
from minihack_env import (CLIFF, ROOM_WITH_MONSTER,
                          get_minihack_envirnment)
from agents import QLearningOffPolicy, DynaQAgent                # <- NEW!

# ─── hyper-parameters ──────────────────────────────────────────
ENV_CLIFF      = CLIFF          # e.g. "MiniHack-Cliff-v0"
ENV_MONSTER    = ROOM_WITH_MONSTER   # e.g. "MiniHack-Room-Monster-5x5-v0"

N_EPIS_CLIFF   = 200
N_EPIS_MONSTER = 15_000
CHUNK          = 1_500               # for histograms
ROLL_W         = 50
SEEDS          = list(range(10))

GAMMA          = 0.99
ALPHA          = 0.15
EPSILON_Q      = 0.10
N_PLANNING     = 10                  # Dyna-Q planning backups

MAX_STEP       = 600                 # per episode
DEATH_PENALTY  = -100
GOOD_WIN_THRES = -10                 # same rule as earlier script

OUTCOMES  = ["good_win", "norm_win", "fail"]
LABEL2IDX = {lbl: i for i, lbl in enumerate(OUTCOMES)}

Metric = collections.namedtuple("Metric",
                                "ret greedy outcomes")          # outcomes==[] for Cliff

# ---------------------------------------------------------------------------
def set_seed(sd: int):
    random.seed(sd);  np.random.seed(sd)

def cum_mean(v): v = np.asarray(v, float); return np.cumsum(v)/np.arange(1,len(v)+1)
def mov_mean(v, w): v = np.asarray(v,float); return np.convolve(v, np.ones(w)/w, mode="valid") if len(v)>=w else np.empty(0)

# ---------------------------------------------------------------------------
def run_one(env_id: str, agent_cls, kw_agent: dict,
            n_episodes: int, seed: int, track_outcomes: bool):
    """Run a single seed, return Metric with episode returns etc."""
    set_seed(seed)
    env = get_minihack_envirnment(env_id, add_pixel=False)
    env.reset(seed=seed)
    agent = agent_cls(id=f"{agent_cls.__name__}-{seed}",
                      action_space=env.action_space, **kw_agent)

    returns, outcomes, chunk_hist = [], [], []
    for ep in range(n_episodes):
        obs, _ = env.reset()
        done, prev_r, steps, G, died = False, 0.0, 0, 0.0, False
        ep_log = []

        while not done and steps < MAX_STEP:
            obs_cp  = copy.deepcopy(obs)
            act     = agent.act(obs, prev_r)
            obs, r, term, trunc, _ = env.step(act)

            if r <= DEATH_PENALTY: died = True
            ep_log.append((obs_cp, act, r))

            G += r; steps += 1; done = term or trunc; prev_r = r

        # per-episode agent hook (base class already has it for Q/DynaQ)
        if hasattr(agent, "onEpisodeEnd"):
            agent.onEpisodeEnd(episode=ep_log, reward=G,
                               episode_number=ep, last_step_reward=prev_r)
        returns.append(G)

        # ----- outcome bookkeeping (Monster only) -----------------------
        if track_outcomes:
            if died or steps >= MAX_STEP:
                label = "fail"
            else:
                label = "good_win" if G > GOOD_WIN_THRES else "norm_win"
            outcomes.append(label)
            if (ep+1) % CHUNK == 0:
                chunk_hist.append(outcomes[-CHUNK:])

    # ----- greedy evaluation -------------------------------------------
    if hasattr(agent, "epsilon"): agent.epsilon = 0.0
    obs, _ = env.reset(); G_eval, steps, done = 0.0, 0, False
    while not done and steps < MAX_STEP:
        obs, r, term, trunc, _ = env.step(agent.act(obs, 0.0))
        G_eval += r; done = term or trunc; steps += 1
    env.close()

    return Metric(np.asarray(returns,float), G_eval, chunk_hist)

# ---------------------------------------------------------------------------
def train_all(env_id, n_episodes, track_outcomes=False):
    """Run both agents for all seeds and return a dict → list[Metric]."""
    records = collections.defaultdict(list)

    CONFIG = {
        "Q-learning": dict(cls=QLearningOffPolicy,
                           kw=dict(epsilon=EPSILON_Q, gamma=GAMMA, alpha=ALPHA)),
        "Dyna-Q"   : dict(cls=DynaQAgent,
                           kw=dict(epsilon=EPSILON_Q, gamma=GAMMA,
                                   alpha=ALPHA, n_planning=N_PLANNING))
    }

    print(f"\n=== Training on {env_id} ===")
    for name, cfg in CONFIG.items():
        for sd in SEEDS:
            print(f"  ▶ {name:<11} seed {sd} … ", end="", flush=True)
            rec = run_one(env_id, cfg["cls"], cfg["kw"],
                          n_episodes, sd, track_outcomes)
            records[name].append(rec)
            print(f"done   (μ_last50 = {rec.ret[-50:].mean():6.1f})")
    return records

# ---------------------------------------------------------------------------
def plot_curves(title, records, roll_w, burn_in=0):
    """Moving-average return across seeds, with ±SEM shading."""
    plt.figure(figsize=(8,4))
    for name, recs in records.items():
        rets = np.stack([r.ret for r in recs])[:, burn_in:]
        mean = rets.mean(axis=0)
        sem  = rets.std(axis=0, ddof=1)/math.sqrt(len(recs))
        x    = np.arange(len(mean))[roll_w-1:]
        plt.plot(x, mov_mean(mean, roll_w), label=name)
        plt.fill_between(x,
                         mov_mean(mean-sem, roll_w),
                         mov_mean(mean+sem, roll_w), alpha=.25)
    plt.title(title)
    plt.xlabel("Episode"); plt.ylabel(f"{roll_w}-ep MA return")
    plt.legend(); plt.tight_layout()

# ---------------------------------------------------------------------------
def monster_histogram(records):
    """Stacked-bar outcome fraction every CHUNK episodes (Monster task)."""
    bins   = np.arange(CHUNK, N_EPIS_MONSTER+1, CHUNK)
    bar_w  = CHUNK * 0.9
    colors = plt.cm.Set2(range(3))

    for name, recs in records.items():
        counts = np.zeros((len(bins), 3), int)
        for rec in recs:
            for k, chunk in enumerate(rec.outcomes):
                for lbl in chunk:
                    counts[k, LABEL2IDX[lbl]] += 1
        frac = counts / (len(recs) * CHUNK)

        plt.figure(figsize=(7,4))
        bottom = np.zeros(len(bins))
        for i, lbl in enumerate(OUTCOMES):
            plt.bar(bins, frac[:, i], width=bar_w, bottom=bottom,
                    color=colors[i], edgecolor="k", linewidth=.4, label=lbl)
            bottom += frac[:, i]

        plt.title(f"{name} — outcome fractions per {CHUNK} episodes")
        plt.ylabel("Fraction"); plt.ylim(0, 1)
        plt.xticks(bins, bins); plt.legend(fontsize=8); plt.tight_layout()

# ---------------------------------------------------------------------------
def summary_table(records, idx_slice):
    table = []
    for name, recs in records.items():
        rets   = np.stack([r.ret for r in recs])
        avg_mu = rets[:, idx_slice].mean(axis=1).mean()
        avg_sem = rets[:, idx_slice].mean(axis=1).std(ddof=1)/math.sqrt(len(recs))
        greedy = np.array([r.greedy for r in recs])
        g_mu, g_sem = greedy.mean(), greedy.std(ddof=1)/math.sqrt(len(recs))
        table.append([name,
                      f"{avg_mu:7.1f} ± {avg_sem:.1f}",
                      f"{g_mu:7.1f} ± {g_sem:.1f}"])
    print(tabulate(table, headers=["Agent", "AvgReturn", "GreedyReturn"],
                   tablefmt="github"))

# ---------------------------------------------------------------------------
if __name__ == "__main__":

    # 1) Room-with-Cliff ---------------------------------------------------
    cliff_rec = train_all(ENV_CLIFF, N_EPIS_CLIFF, track_outcomes=False)
    plot_curves("Room-with-Cliff — 20-ep MA return",
                cliff_rec, ROLL_W, burn_in=0)
    summary_table(cliff_rec, idx_slice=slice(200, None))

    # # 2) Room-with-Monster -------------------------------------------------
    # mon_rec   = train_all(ENV_MONSTER, N_EPIS_MONSTER, track_outcomes=True)
    # plot_curves("Room-with-Monster — 50-ep MA return",
    #             mon_rec, ROLL_W, burn_in=200)            # ignore first 200 for smoother curve
    # summary_table(mon_rec, idx_slice=slice(200, None))
    # monster_histogram(mon_rec)

    # plt.show()
