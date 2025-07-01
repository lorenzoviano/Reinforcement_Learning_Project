# test_fixedagent.py
import minihack_env
import matplotlib.pyplot as plt
# remove/import animation if you like, but it won’t be used
# import matplotlib.animation as animation
from agents import FixedAgent
from commons import get_crop_pixel_from_observation

# Map NLE’s action indices to human names
action_names = {0: "UP", 1: "RIGHT", 2: "DOWN", 3: "LEFT"}

for env_id in (minihack_env.EMPTY_ROOM, minihack_env.ROOM_WITH_LAVA):
    print(f"\n=== Testing FixedAgent on {env_id} ===\n")
    env = minihack_env.get_minihack_envirnment(env_id, add_pixel=True)
    agent = FixedAgent("fixed", env.action_space)

    obs, info = env.reset()
    frames, annotations = [], []

    for t in range(10):
        action = agent.act(obs)
        obs, reward, done, truncated, info = env.step(action)

        # 🛠 Debug‐annotation: only action & reward
        annotations.append(f"Step {t}: {action_names[action]}, reward={reward}")

        # Crop the pixel observation and save it
        frames.append(get_crop_pixel_from_observation(obs))

        if done or truncated:
            break

    env.close()

    # --- REPLACED: show each frame as its own static image ---
    for i, (frame, anno) in enumerate(zip(frames, annotations)):
        plt.figure(figsize=(4, 4))
        plt.imshow(frame)
        plt.axis("off")
        plt.show()
