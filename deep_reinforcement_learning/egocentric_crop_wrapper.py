import gymnasium as gym
import numpy as np

class EgocentricCropWrapper(gym.ObservationWrapper):
    """
    Crops a square window of size (crop_size x crop_size) centered around
    the agent's position in the 'chars' observation grid.
    Flattens the resulting crop into a 1D numpy array.
    """
    def __init__(self, env, crop_size=9):
        super().__init__(env)
        assert hasattr(env.observation_space, 'spaces') and 'chars' in env.observation_space.spaces, \
            "Env must have a 'chars' observation channel"

        self.crop_size = crop_size
        self.orig_shape = env.observation_space.spaces['chars'].shape
        # New observation: flattened crop of chars
        flat_size = crop_size * crop_size
        self.observation_space = gym.spaces.Box(
            low=0,
            high=self.orig_shape[0] - 1,
            shape=(flat_size,),
            dtype=np.int32
        )

    def observation(self, obs):
        # obs is a dict with keys like 'chars', 'pixels', etc.
        chars = obs['chars']
        # find agent location (glyph code 64 '@')
        agent_pos = np.argwhere(chars == ord('@'))
        if len(agent_pos) == 0:
            # fallback: treat center as agent
            cy, cx = self.orig_shape[0] // 2, self.orig_shape[1] // 2
        else:
            cy, cx = agent_pos[0]

        half = self.crop_size // 2
        # compute crop bounds
        y0 = max(cy - half, 0)
        x0 = max(cx - half, 0)
        y1 = min(cy + half + 1, self.orig_shape[0])
        x1 = min(cx + half + 1, self.orig_shape[1])

        # extract crop, pad if necessary
        crop = np.zeros((self.crop_size, self.crop_size), dtype=chars.dtype)
        crop_y0 = half - (cy - y0)
        crop_x0 = half - (cx - x0)
        crop[crop_y0:crop_y0 + (y1 - y0), crop_x0:crop_x0 + (x1 - x0)] = chars[y0:y1, x0:x1]

        # flatten
        flat = crop.flatten()
        return flat
