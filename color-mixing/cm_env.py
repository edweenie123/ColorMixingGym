import gym
from gym import spaces
import numpy as np

from stable_baselines3 import DDPG
from stable_baselines3.common.noise import NormalActionNoise

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
    def __init__(self, available_paints: List[Paint], target_color: Tuple[int, int, int], max_steps: int = 100):
        self.available_paints = available_paints
        self.target_paint = Paint(target_color, 0)  # Target color with 0 amount
        self.current_paint = Paint((255, 255, 255), 0)  # Starting with no paint
        self.max_steps = max_steps
        self.step_count = 0

        # Define action and observation spaces
        num_paints = len(available_paints)
        self.action_space = spaces.Tuple((spaces.Discrete(num_paints), spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)))
        self.observation_space = spaces.Box(low=0, high=255, shape=(3 + num_paints,), dtype=np.float32)

    def reset(self) -> np.ndarray:
        self.current_paint = Paint((255, 255, 255), 0)
        self.remaining_paints = [Paint(paint.color, paint.amount) for paint in self.available_paints]
        self.step_count = 0
        return self._get_observation()

    def step(self, action: Tuple[int, float]) -> Tuple[np.ndarray, float, bool, dict]:
        paint_index, amount_ratio = action
        selected_paint = self.remaining_paints[paint_index]

        # TODO: need to add noise => add some stochasticity to amount
        # Calculate the actual amount to mix based on the ratio
        amount_to_mix = amount_ratio * selected_paint.amount

        # Mix the selected paint with the current color
        paint_to_mix, _ = selected_paint.split(amount_to_mix)
        self.current_paint = self.current_paint.mix_with(paint_to_mix)

        # Update the environment for the next step
        self.step_count += 1
        done = self.step_count >= self.max_steps
        reward = self._calculate_reward()
        return self._get_observation(), reward, done, {}

    def _calculate_reward(self) -> float:
        """
        Use negative Euclidean distance
        """
        distance = np.linalg.norm(np.array(self.current_paint.color) - np.array(self.target_paint.color))
        return -distance

    def _get_observation(self) -> np.ndarray:
        remaining_amounts = [paint.amount for paint in self.remaining_paints]
        return np.concatenate([np.array(self.current_paint.color), remaining_amounts])

# Example setup
available_paints = [
    Paint((255, 0, 0), 100),
    Paint((0, 255, 0), 100),
    Paint((0, 0, 255), 100)
]

target_color = [128, 128, 0]  # Example target color (olive green)
# env = ColorMixingEnv(available_paints, target_color)
