import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random

class LivestockMonitoringEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 5}

    def __init__(self):
        super().__init__()
        self.grid_size = 10
        self.num_animals = 5
        self.max_steps = 200  # shorter episodes

        # Actions: 0=up, 1=down, 2=left, 3=right
        self.action_space = spaces.Discrete(4)

        # Observation space: drone (x,y) + for each animal (x,y, distress flag)
        obs_len = 2 + self.num_animals * 3
        self.observation_space = spaces.Box(
            low=0,
            high=self.grid_size - 1,
            shape=(obs_len,),
            dtype=np.int32
        )

        self.reset()

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self.drone_pos = [0, 0]
        self.steps = 0

        self.animals = []
        for _ in range(self.num_animals):
            x, y = random.randint(0, 9), random.randint(0, 9)
            self.animals.append({"pos": [x, y], "distress": False})

        self.trigger_health_event()
        return self._get_obs(), {}

    def trigger_health_event(self):
        idx = random.randint(0, self.num_animals - 1)
        self.animals[idx]["distress"] = True

    def _get_obs(self):
        obs = []
        obs.extend(self.drone_pos)
        for a in self.animals:
            obs.extend(a["pos"])
            obs.append(int(a["distress"]))
        return np.array(obs, dtype=np.int32)

    def step(self, action):
        # Move drone
        if action == 0 and self.drone_pos[1] > 0:
            self.drone_pos[1] -= 1
        elif action == 1 and self.drone_pos[1] < self.grid_size - 1:
            self.drone_pos[1] += 1
        elif action == 2 and self.drone_pos[0] > 0:
            self.drone_pos[0] -= 1
        elif action == 3 and self.drone_pos[0] < self.grid_size - 1:
            self.drone_pos[0] += 1

        # Move animals
        for a in self.animals:
            if a["distress"]:
                # 30% chance to move if distressed
                if random.random() < 0.3:
                    dx, dy = random.choice([(0, 1), (0, -1), (1, 0), (-1, 0), (0, 0)])
                else:
                    dx, dy = (0, 0)
            else:
                dx, dy = random.choice([(0, 1), (0, -1), (1, 0), (-1, 0), (0, 0)])

            new_x = min(max(a["pos"][0] + dx, 0), self.grid_size - 1)
            new_y = min(max(a["pos"][1] + dy, 0), self.grid_size - 1)
            a["pos"] = [new_x, new_y]

        reward = -0.1  # step penalty to reduce idle wandering
        detected = False

        # Check if drone detects any distressed animal
        for a in self.animals:
            if a["distress"] and a["pos"] == self.drone_pos:
                reward += 20  # reward for successful detection
                a["distress"] = False
                detected = True

        if not detected:
            # Encourage getting closer to distressed animal(s)
            distressed_animals = [a for a in self.animals if a["distress"]]
            if distressed_animals:
                distances = [
                    abs(self.drone_pos[0] - a["pos"][0]) + abs(self.drone_pos[1] - a["pos"][1])
                    for a in distressed_animals
                ]
                min_dist = min(distances)
                reward += max(0, 1 - 0.1 * min_dist)  # higher reward if closer

        self.steps += 1
        terminated = all(not a["distress"] for a in self.animals)
        truncated = self.steps >= self.max_steps

        return self._get_obs(), reward, terminated, truncated, {}

    def render(self, mode="human"):
        # Rendering handled externally (e.g., in Pygame)
        pass
