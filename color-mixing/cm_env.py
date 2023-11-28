import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pygame
from typing import Tuple, Optional, List
import random


# helper functions for subtractive color mixing
class SubtractiveModel:
    @staticmethod
    def _rgb_to_cmy(rgb: Tuple[int, int, int]) -> Tuple[int, int, int]:
        return tuple(255 - value for value in rgb)

    @staticmethod
    def _cmy_to_rgb(cmy: Tuple[int, int, int]) -> Tuple[int, int, int]:
        return tuple(255 - value for value in cmy)

    @staticmethod
    def mix_colors(paint1: 'Paint', paint2: 'Paint') -> Tuple[int, int, int]:
        cmy1 = SubtractiveModel._rgb_to_cmy(paint1.color)
        cmy2 = SubtractiveModel._rgb_to_cmy(paint2.color)
        total_amount = paint1.amount + paint2.amount
        if paint1.amount == 0:
            return paint2.color
        if paint2.amount == 0:
            return paint1.color

        mixed_cmy = tuple(int((cmy1[i] * paint1.amount + cmy2[i] * paint2.amount) / total_amount) for i in range(3))
        return SubtractiveModel._cmy_to_rgb(mixed_cmy)

class Paint:
    def __init__(self, color: Tuple[int, int, int], amount: float):
        self.color = color      # color in RGB space
        self.amount = amount    # amount (in ml)

    def mix_with(self, other: 'Paint') -> 'Paint':
        mixed_amount = self.amount + other.amount
        mixed_color = SubtractiveModel.mix_colors(self, other)
        return Paint(mixed_color, mixed_amount)

    def subtract_amount(self, amount: float) -> None:
        self.amount = max(self.amount - amount, 0)

    def split(self, split_amount: int) -> 'Paint':
        """
        Split a portion of the paint into a new Paint object.
        """
        assert(split_amount >= 0)
        assert(split_amount <= self.amount)
        # if split_amount <= 0:
        #     return Paint(self.color, 0)  # No paint to split

        # if split_amount > self.amount:
        #     self.amount = 0
        #     return Paint(self.color, self.amount)  # Split all paint

        # Create new Paint object with the split amount
        new_paint = Paint(self.color, split_amount)

        # Subtract the split amount from the current paint
        self.subtract_amount(split_amount)

        # Return the new paint and the updated current paint
        return new_paint

class ColorMixingEnv(gym.Env):
    def __init__(self, num_beakers, max_steps: int = 10, noise_level=0):

        # randomly initialize paints for beakrs  
        self.max_steps = max_steps
        self.reset()
        # self.beakers = beakers
        # self.target_paint = Paint(target_color, target_amount)
        # self.step_count = 0
        # self.done = False

        # num_beakers = len(beakers)
        self.action_space = spaces.MultiDiscrete([
            num_beakers,    # From beaker (index)
            num_beakers,    # To beaker (index)
            100,             # 100 possible transfer amounts
            2               # Binary action for 'done'
        ])  
        self.observation_space = spaces.Box(
            low=0, high=255, 
            shape=(num_beakers + 1, 4), 
            dtype=np.int64
        )

        self.initial_score = self.calculate_score()  
        
        self.noise_level = noise_level

    def reset(self, **kwargs) -> np.ndarray:
        # ~ should reset generate a random new state?
        # Reset logic
        # self.beakers = [Paint(paint.color, paint.amount) for paint in self.beakers]

        self.beakers=[
            Paint((255, 0, 0), 100),
            Paint((0, 255, 0), 100),
            Paint((0, 0, 255), 100),
            Paint((0, 0, 0), 0) # empty beaker
        ]
        
        # Randomize the contents of each beaker
        # for beaker in self.beakers:
        #     beaker.color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        #     beaker.amount = random.randint(40, 120)  # max_amount is the maximum paint amount in a beaker

        # self.beakers=[
        #     Paint((255, 0, 0), 100),
        #     Paint((0, 255, 0), 100),
        #     Paint((0, 0, 255), 100),
        #     Paint((0, 0, 0), 0) # empty beaker
        # ]

        # make last beaker empty 
        self.beakers[-1].amount = 0

        # Randomize the target color and amount
        rand_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        rand_amount = random.randint(40, 150) # Set suitable min and max amounts
        self.target_paint = Paint(rand_color, rand_amount)

        self.step_count = 0
        self.initial_score = self.calculate_score()
        self.previous_action = None

        
        # Return the initial observation
        return self._get_observation(), {}

    def step(self, action: Tuple[int, int, int, int]) -> Tuple[np.ndarray, float, bool, dict]:
        # TODO: need to add stochasticity 
        from_beaker_index, to_beaker_index, amount, done_action = action

        self.previous_action = (from_beaker_index, to_beaker_index, amount, done_action)

        # Define the noise level as a percentage of the amount
        noise = np.random.normal(0, self.noise_level * amount)


        from_beaker = self.beakers[from_beaker_index]
        to_beaker = self.beakers[to_beaker_index]

        # Apply noise to the amount to transfer
        amount_to_transfer = min(from_beaker.amount, max(amount + noise, 0)) 
        paint_to_transfer = from_beaker.split(amount_to_transfer)

        mixed_paint = to_beaker.mix_with(paint_to_transfer)
        self.beakers[to_beaker_index] = mixed_paint

        self.step_count += 1
        done = (self.step_count >= self.max_steps) or done_action
        reward = self._calculate_reward(done=done)
        
        truncated = False # ~ don't know what the set this
        return self._get_observation(), reward, done, truncated, {}
    
    def calculate_score(self) -> float:
        # ~ this is a very myopic reward...
        # among all the beakers, we choose the one with the lowest "cost"
        # cost = color difference + amount difference 
        # color differience is use Euclidiean distance in RGB space

        max_color_distance = np.linalg.norm(np.array([255, 255, 255]))  # Maximum Euclidean distance in RGB space
        max_amount_distance = 200  # Replace with your value

        color_weight = 0.8  # Weight for color distance, e.g., 0.5
        amount_weight = 0.2  # Weight for amount distance, e.g., 0.5


        beaker = self.beakers[-1]

        color_distance = np.linalg.norm(np.array(beaker.color) - np.array(self.target_paint.color)) / max_color_distance

        if (beaker.amount == 0): # when amount is 0, color of paint is meaningless
            color_distance = 1
        
        # Normalize the amount distance
        amount_distance = abs(beaker.amount - self.target_paint.amount) / max_amount_distance

        # Apply weights to color and amount distances
        total_distance = color_weight * color_distance + amount_weight * amount_distance
        
        # Invert the distance to make the reward positive
        score = (1 - total_distance) 

        def weighting_function(x : float) -> float:
            return x**4
        
        score = weighting_function(score)

        return score 
    
    def _calculate_reward(self, done=False) -> float:
        if done:
            current_score = self.calculate_score()
            # improvement = current_score - self.initial_score
            # reward = max(improvement, 0)  
            return current_score

        # Update the best heuristic score
        # if current_score > self.best_heuristic_score:
        #     self.best_heuristic_score = current_score
        
        step_penalty = 0.01
        reward = -step_penalty

        return reward

    def _get_observation(self) -> np.ndarray:
        colors = [np.array(beaker.color) for beaker in self.beakers]
        colors.append(np.array(self.target_paint.color))
        colors = np.array(colors)  # 2D array of shape (num_beakers, 3)
        amounts = [[beaker.amount] for beaker in self.beakers]
        amounts.append([self.target_paint.amount])
        amounts = np.array(amounts)  # 2D array of shape (num_beakers, 1)
        # print('colors', colors)
        # print('amounts', amounts)
        # print((np.concatenate([colors, amounts], axis=1)[0][0]).dtype)

        return np.concatenate([colors, amounts], axis=1).astype('int64')  # Concatenate along columns

    def load_state_from_file(self, file_path):
        with open(file_path, 'r') as file:
            self.beakers = []
            self.target_paint = None

            for line in file:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue  # Skip invalid lines

                line_type, red, green, blue, amount = parts
                color = (int(red), int(green), int(blue))
                amount = float(amount)

                if line_type == 'B':
                    self.beakers.append(Paint(color, amount))
                elif line_type == 'T':
                    self.target_paint = Paint(color, amount)

        # Reset other necessary environment variables
        self.step_count = 0
        self.previous_action = None

        # Return the initial observation
        return self._get_observation()

    def render(self, mode='human'):
        # Initialize Pygame if it hasn't been initialized
        if not pygame.get_init():
            pygame.init()

        # Define window dimensions and colors
        screen_width = 800
        screen_height = 400
        background_color = (255, 255, 255)  # White background

        # Create Pygame screen
        screen = pygame.display.set_mode((screen_width, screen_height))
        screen.fill(background_color)

        # Define beaker dimensions and positions
        beaker_width = 60  # 40 pixels
        beaker_height = 100  # Height of the beaker in pixels
        space_between_beakers = 20  # 10 pixels
        beaker_capacity = 200  # Max capacity of each beaker (in units, not pixels)
        total_width = beaker_width + space_between_beakers
        beaker_border_width = 2  # Thickness of beaker border
        y_pad = 50  # Padding from the top of the screen
        target_offset = 30  # Additional space before the target beaker


        # Centering the beakers
        total_beakers_width = (len(self.beakers) + 1) * total_width  # Including target beaker
        start_x = (screen_width - total_beakers_width) // 2
        
        # Font for labels
        font = pygame.font.SysFont(None, 24)
        amount_font = pygame.font.SysFont(None, 20)  # Font for displaying the amount


        # Draw each beaker and label
        for i, beaker in enumerate(self.beakers):
            x_position = start_x + i * total_width

            # Fill beaker with paint
            paint_height = beaker.amount / beaker_capacity * beaker_height
            pygame.draw.rect(screen, beaker.color, (x_position, screen_height - y_pad - paint_height, beaker_width, paint_height))

            # Draw beaker border 
            pygame.draw.rect(screen, (0, 0, 0), (x_position, screen_height - y_pad - beaker_height, beaker_width, beaker_height), beaker_border_width)

            # Add label
            label = font.render(str(i), True, (0, 0, 0))
            screen.blit(label, (x_position + beaker_width // 2 - label.get_width() // 2, screen_height - y_pad + 5))
            
            # Add label for beaker amount
            amount_label = amount_font.render(f"{beaker.amount:.0f}", True, (0, 0, 0))
            screen.blit(amount_label, (x_position + beaker_width // 2 - amount_label.get_width() // 2, screen_height - y_pad - beaker_height - 20))

        # Draw the target beaker (further to the right)
        target_x_position = start_x + len(self.beakers) * total_width + target_offset
        paint_height = self.target_paint.amount / beaker_capacity * beaker_height
        pygame.draw.rect(screen, self.target_paint.color, (target_x_position, screen_height - y_pad - paint_height, beaker_width, paint_height))
        pygame.draw.rect(screen, (0, 0, 0), (target_x_position, screen_height - y_pad - beaker_height, beaker_width, beaker_height), beaker_border_width)

        # Add text under the target beaker
        text = font.render('Target', True, (0, 0, 0))
        screen.blit(text, (target_x_position + beaker_width // 2 - text.get_width() // 2, screen_height - y_pad + 5))
        
        # Add label for target beaker amount
        target_amount_label = amount_font.render(f"{self.target_paint.amount:.0f}", True, (0, 0, 0))
        screen.blit(target_amount_label, (target_x_position + beaker_width // 2 - target_amount_label.get_width() // 2, screen_height - y_pad - beaker_height - 20))


        # visualize previous action
        if self.previous_action is not None:
            from_idx, to_idx, amount, done_action = self.previous_action

            if done_action:
                done_font = pygame.font.SysFont(None, 48)  # Larger font for "Done" text
                done_text = done_font.render("Done", True, (0, 0, 0))
                screen.blit(done_text, (screen_width // 2 - done_text.get_width() // 2, 10))  # Top center position
            
            # Calculate positions for the start and end of the arrow
            from_x = start_x + from_idx * total_width + beaker_width / 2
            to_x = start_x + to_idx * total_width + beaker_width / 2
            y = screen_height - y_pad - beaker_height / 2 - 100

            # Draw the arrow
            pygame.draw.line(screen, (0, 0, 0), (from_x, y), (to_x, y), 3)
            pygame.draw.polygon(screen, (0, 0, 0), [(to_x, y), (to_x - 10, y - 5), (to_x - 10, y + 5)])

            # Add label for the transfer amount
            transfer_label = amount_font.render(f"{amount:.0f}", True, (0, 0, 0))
            mid_x = (from_x + to_x) / 2
            screen.blit(transfer_label, (mid_x - transfer_label.get_width() / 2, y - 20))

        # Update the display
        pygame.display.flip()

        # Keep the window open until clicked
        wait_for_click = True
        while wait_for_click:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    wait_for_click = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    wait_for_click = False

if __name__ == '__main__':

    # Create the environment
    env = ColorMixingEnv(
        beakers=[
            Paint((255, 0, 0), 100),
            Paint((0, 255, 0), 100),
            Paint((0, 0, 255), 100),
            Paint((0, 0, 0), 0) # empty beaker
        ],
        target_color=(128, 128, 0),  # Olive green
        target_amount=150
    )

    # env.render()
    print('initial score:', env.calculate_score())
    # input()
    env.render()
    obs, reward, done, trunacated, _ = env.step((0, 3, 75, 0))
    print('step 1:', obs, reward, env.calculate_score())
    env.render()

    obs, reward, done, trunacated, _ = env.step((1, 3, 75, 0), )

    print('step 2:', obs, reward, env.calculate_score())
    env.render()




