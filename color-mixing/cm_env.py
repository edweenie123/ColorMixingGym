import gymnasium as gym
from gymnasium import spaces
import numpy as np

# from stable_baselines3 import DDPG
# from stable_baselines3.common.noise import NormalActionNoise

from typing import Tuple, Optional, List

# helper functions for subtractive color mixing
class SubtractiveModel:
    @staticmethod
    def _rgb_to_cmy(rgb: Tuple[int, int, int]) -> Tuple[int, int, int]:
        return tuple(255 - value for value in rgb)

    @staticmethod
    def _cmy_to_rgb(cmy: Tuple[int, int, int]) -> Tuple[int, int, int]:
        return tuple(255 - value for value in cmy)

    @staticmethod
    def mix_colors(paint1: 'Paint', paint2: 'Paint') -> 'Paint':
        cmy1 = SubtractiveModel._rgb_to_cmy(paint1.color)
        cmy2 = SubtractiveModel._rgb_to_cmy(paint2.color)
        total_amount = paint1.amount + paint2.amount
        mixed_cmy = tuple(int((cmy1[i] * paint1.amount + cmy2[i] * paint2.amount) / total_amount) for i in range(3))
        return SubtractiveModel._cmy_to_rgb(mixed_cmy)

class Paint:
    def __init__(self, color: Tuple[int, int, int], amount: float):
        self.color = color      # color in RGB space
        self.amount = amount    # amount (in ml)

    def mix_with(self, other: 'Paint') -> 'Paint':
        mixed_amount = self.amount + other.amount
        mixed_color = SubtractiveModel.subtractive_mix_colors(self, other)
        return Paint(mixed_color.rgb, mixed_amount)

    def subtract_amount(self, amount: float) -> None:
        self.amount = max(self.amount - amount, 0)

    def split(self, split_amount: float) -> Tuple['Paint', Optional['Paint']]:
        """
        Split a portion of the paint into a new Paint object.
        """
        if split_amount <= 0:
            return Paint(self.color, 0), None  # No paint to split

        if split_amount >= self.amount:
            return Paint(self.color, self.amount), None  # Split all paint

        # Create new Paint object with the split amount
        new_paint = Paint(self.color, split_amount)

        # Subtract the split amount from the current paint
        self.subtract_amount(split_amount)

        # Return the new paint and the updated current paint
        return new_paint, Paint(self.color, self.amount)

class ColorMixingEnv(gym.Env):
    def __init__(self, beakers: List[Paint], target_color: Tuple[int, int, int], target_amount: float, max_steps: int = 100):
        self.beakers = beakers
        
        # ~specify both target color AND target amount
        self.target_paint = Paint(target_color, target_amount)
        self.max_steps = max_steps
        self.step_count = 0

        num_beakers = len(beakers)
        self.action_space = spaces.Tuple((
            spaces.Discrete(num_beakers),  # From beaker (index)
            spaces.Discrete(num_beakers),  # To beaker (index)
            spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)  # Amount ratio
        ))
        self.observation_space = spaces.Box(low=0, high=255, shape=(3 * num_beakers + num_beakers,), dtype=np.float32)

    def reset(self) -> np.ndarray:
        self.beakers = [Paint(paint.color, paint.amount) for paint in self.beakers]
        self.step_count = 0
        return self._get_observation()

    def step(self, action: Tuple[int, int, float]) -> Tuple[np.ndarray, float, bool, dict]:
        # TODO: need to add stochasticity 

        from_beaker_index, to_beaker_index, amount_ratio = action
        from_beaker = self.beakers[from_beaker_index]
        to_beaker = self.beakers[to_beaker_index]

        amount_to_transfer = amount_ratio * from_beaker.amount
        paint_to_transfer, _ = from_beaker.split(amount_to_transfer)
        mixed_paint = to_beaker.mix_with(paint_to_transfer)
        self.beakers[to_beaker_index] = mixed_paint

        self.step_count += 1
        done = self.step_count >= self.max_steps
        reward = self._calculate_reward()
        return self._get_observation(), reward, done, {}

    def _calculate_reward(self) -> float:
        # ~ this is a very myopic reward...
        # among all the beakers, we choose the one with the lowest "cost"
        # cost = color difference + amount difference 
        # color differience is use Euclidiean distance in RGB space

        best_distance = float('inf')
        for beaker in self.beakers:
            color_distance = np.linalg.norm(np.array(beaker.color) - np.array(self.target_paint.color))
            amount_distance = abs(beaker.amount - self.target_paint.amount)
            total_distance = color_distance + amount_distance
            if total_distance < best_distance:
                best_distance = total_distance
        return -best_distance

    def _get_observation(self) -> np.ndarray:
        colors = [np.array(beaker.color) for beaker in self.beakers]
        amounts = [beaker.amount for beaker in self.beakers]
        return np.concatenate(colors + amounts)

beakers = [
    Paint((255, 0, 0), 100),
    Paint((0, 255, 0), 100),
    Paint((0, 0, 255), 100)
]

target_color = (128, 128, 0)  # Example target color (olive green)
target_amount = 150  # Example target amount

env = ColorMixingEnv(beakers, target_color, target_amount)
