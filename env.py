import gymnasium as gym
from gymnasium import spaces
import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.patches as patches
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

        # Create new Paint object with the split amount
        new_paint = Paint(self.color, split_amount)

        # Subtract the split amount from the current paint
        self.subtract_amount(split_amount)

        # Return the new paint and the updated current paint
        return new_paint

class ColorMixingEnv(gym.Env):
    def __init__(self, num_beakers: int = 4, max_steps: int = 10, noise_level=0):

        # randomly initialize paints for beakrs  
        self.max_steps = max_steps
        self.beaker_capacity = 200 # constant capacity limit for now

        self.set_num_beakers(num_beakers, load_dummy=True)
        self.cmp_beaker = 0
        # self.rand_mode = rand_mode
        # self.initial_score = self.calculate_score()  
        self.noise_level = noise_level
        # self.initialized_screen
        self.screen = None
        self.reset()
    
    def set_num_beakers(self, num_beakers, load_dummy=False):
        self.num_beakers = num_beakers

        # initialize dummy state
        if load_dummy:
            self.beakers = []
            for _ in range(self.num_beakers):
                self.beakers.append(Paint((0, 0, 0), 0))
            self.target_beaker = Paint((0, 0, 0), 0) 

        self.action_space = spaces.MultiDiscrete([
            num_beakers,      # From beaker (index)
            num_beakers,      # To beaker (index)
            100,              # 100 possible transfer amounts
            2,                # Binary action for 'done'
            num_beakers       # Beaker index for comparison (only relevant if 'done') 
        ])  

        self.observation_space = spaces.Box(
            low=0, high=255, 
            shape=(num_beakers + 1, 4), 
            dtype=np.int64
        )


    def reset(self, **kwargs) -> np.ndarray:
        
        self._randomize_state(mode=3)

        self.step_count = 0
        self.prev_action = None
        
        # Return the initial observation
        return self._get_observation(), {}
    
    def _randomize_state(self, mode=0):
        """
        Randomizes the state of the beakers based on the specified mode.

        Parameters:
        mode (int): Determines the randomization approach. Default is 0.

        Modes:
            0 - Fully Randomized State (this is very difficult)
                - Randomizes the color and amount of paint in all beakers.
                - Randomizes the target beaker's color and amount.

            1 - Guarantees noisy CMY in 3 beakers
                - Initializes 3 beakers to be cyan, megenta, and yellow
                    - However, amounts and exact shades will be subject to noise 
                - One other beaker will have random color and amount.
                - All other beakers will be empty
                - Randomly shuffle the order of the beakers
                - Randomizes the target beaker's color and amount.

            2 - Same thing as 1, but with no noise and no shuffling
            3 - Fixed state for testing
        """

        if mode == 0:
            # Randomize the contents of each beaker
            for beaker in self.beakers:
                beaker.color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                beaker.amount = random.randint(40, 120)  # max_amount is the maximum paint amount in a beaker
        elif mode == 1:
            cmy = [
                (0, 255, 255),  # cyan
                (255, 0, 255),  # megenta
                (255, 255, 0)   # yellow
            ]


            beakers = []

            # apply noise to each primary color 
            for col in cmy:
                noisy_col = list(col)
                for i in range(3):
                    noise = random.randint(-10, 10)
                    noisy_col[i] = int(np.clip(col[i] + noise, 0, 255))
                rand_amt = random.randint(70, 120)
                beakers.append(Paint(tuple(noisy_col), rand_amt))
            
            rand_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            rand_amount = random.randint(40, 120)
            rand_beaker = Paint(rand_color, rand_amount)
            
            beakers.append(rand_beaker)

            # set the rest to be empty beakers
            for _ in range(self.num_beakers - 4):
                beakers.append(Paint((0, 0, 0), 0))
            
            random.shuffle(beakers)

            self.beakers = beakers
        elif mode == 2:
            # cmy = [
            #     (0, 255, 255),  # cyan
            #     (255, 0, 255),  # megenta
            #     (255, 255, 0)   # yellow
            # ]
            cmy = [
                (255, 0, 0),  # cyan
                (0, 255, 0),  # megenta
                (0, 0, 255)   # yellow
            ]

            beakers = []

            # apply noise to each primary color 
            for col in cmy:
                noisy_col = list(col)
                for i in range(3):
                    noise = 0
                    noisy_col[i] = int(np.clip(col[i] + noise, 0, 255))
                rand_amt = random.randint(70, 120)
                beakers.append(Paint(tuple(noisy_col), rand_amt))
            
            # rand_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            # rand_amount = random.randint(40, 120)
            # rand_beaker = Paint(rand_color, rand_amount)
            
            # beakers.append(rand_beaker)

            # set the rest to be empty beakers
            for _ in range(self.num_beakers - 3):
                beakers.append(Paint((0, 0, 0), 0))
            
            # random.shuffle(beakers)

            self.beakers = beakers
        elif mode == 3:
            cmy = [
                (255, 0, 0),  # cyan
                (0, 255, 0),  # megenta
                (0, 0, 255)   # yellow
            ]

            beakers = []

            # apply noise to each primary color 
            for col in cmy:
                beakers.append(Paint(col, 100))
            
            # rand_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            # rand_amount = random.randint(40, 120)
            # rand_beaker = Paint(rand_color, rand_amount)
            
            # beakers.append(rand_beaker)

            # set the rest to be empty beakers
            for _ in range(self.num_beakers - 3):
                beakers.append(Paint((0, 0, 0), 0))
            
            # random.shuffle(beakers)

            self.beakers = beakers

            rand_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            rand_amount = random.randint(40, 120) # set suitable min and max amounts
            self.target_beaker = Paint((198, 218, 116), 67)
            
            return
            
        else:
            raise Exception('invalid randomization mode')
        
        # randomize the target color and amount
        rand_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        rand_amount = random.randint(40, 120) # set suitable min and max amounts
        self.target_beaker = Paint(rand_color, rand_amount)

    def step(self, action: Tuple[int, int, int, int]) -> Tuple[np.ndarray, float, bool, dict]:
        from_idx, to_idx, amt, is_done, cmp_idx = action

        # make std of noise proportional to noise_level and amt 
        noise = np.random.normal(0, self.noise_level * amt)

        from_beaker = self.beakers[from_idx]
        to_beaker = self.beakers[to_idx]

        # apply noise to the amount to transfer
        noisy_amt = max(int(amt + noise), 0)
        
        space_left = self.beaker_capacity - to_beaker.amount
        actual_amt = min(from_beaker.amount, noisy_amt) # can take at most from_beaker.amout
        actual_amt = min(space_left, actual_amt) # can transfer at most space_left 
        paint_to_transfer = from_beaker.split(actual_amt)

        mixed_paint = to_beaker.mix_with(paint_to_transfer)
        self.beakers[to_idx] = mixed_paint

        self.step_count += 1
        done = (self.step_count >= self.max_steps) or is_done
        reward = self._calculate_reward(cmp_idx, done=done)

        # store previous action for rendering 
        self.prev_action = {
            'from_idx' : from_idx, 
            'to_idx' : to_idx, 
            'desired_amt': amt, 
            'actual_amt': actual_amt, 
            'is_done': done,
            'cmp_idx': cmp_idx 
        }
        
        self.cmp_beaker = cmp_idx # ~ this is not meaningful if is_done = False

        truncated = False # ~ don't know what the set this
        return self._get_observation(), reward, done, truncated, {}
    
    def calculate_scores(self) -> float:
        max_color_distance = np.linalg.norm(np.array([255, 255, 255]))  # Maximum Euclidean distance in RGB space
        
        # ! need to set this to max beaker capacity...
        max_amount_distance = 200  

        cmp_beaker = self.beakers[self.cmp_beaker]

        color_distance = np.linalg.norm(np.array(cmp_beaker.color) - np.array(self.target_beaker.color)) / max_color_distance

        if (cmp_beaker.amount == 0): # when amount is 0, color of paint is meaningless
            color_distance = 1
        
        # Normalize the amount distance
        amount_distance = abs(cmp_beaker.amount - self.target_beaker.amount) / max_amount_distance
        
        return 1 - color_distance, 1 - amount_distance
    
    def calculate_weighted_score(self) -> float:
        """
        Calculates a score from 0 - 1 based on the similarity between the 
        properties of a selected beaker and the target beaker.

        The score is derived from the weighted sum of the Euclidean distance in 
        RGB color space and the difference in paint amount, normalized and 
        transformed by a weighting function.

        Returns:
        float: The calculated score, representing the similarity between the selected beaker
        """
        
        color_score, amount_score = self.calculate_scores()

        color_weight = 0.9  # weight for color distance
        amount_weight = 0.1  # weight for amount distance

        # Apply weights to color and amount distances
        score = color_weight * color_score + amount_weight * amount_score
        
        # Invert the distance to make the reward positive
        # score = (1 - total_distance) 
        def weighting_function(x : float) -> float:
            return x**4
        
        score = weighting_function(score)

        return score 
    
    def _calculate_reward(self, cmp_beaker_idx: int, done=False) -> float:
        step_penalty = 0.01
        reward = -step_penalty

        if done:
            reward += self.calculate_weighted_score()

        return reward

    def _get_observation(self) -> np.ndarray:
        colors = [np.array(beaker.color) for beaker in self.beakers]
        colors.append(np.array(self.target_beaker.color))
        colors = np.array(colors)  # 2D array of shape (num_beakers, 3)
        amounts = [[beaker.amount] for beaker in self.beakers]
        amounts.append([self.target_beaker.amount])
        amounts = np.array(amounts)  # 2D array of shape (num_beakers, 1)
        # print('colors', colors)
        # print('amounts', amounts)
        # print((np.concatenate([colors, amounts], axis=1)[0][0]).dtype)

        return np.concatenate([colors, amounts], axis=1).astype('int64')  # Concatenate along columns

    def get_env_text_repr(self):
        """
        Receives observation from environment (env state) and converts it
        to a string representation
        """
        
        obs = self._get_observation()
        beaker_descriptions = []
        num_beakers = len(obs) - 1  # Excluding the target beaker

        # Process each beaker
        for idx in range(num_beakers):
            beaker = obs[idx]
            color_str = f"Color: RGB({beaker[0]}, {beaker[1]}, {beaker[2]})"
            beaker_descriptions.append(f"Beaker {idx}: {color_str}, amount: {beaker[3]}ml")

        # Process the target beaker
        target_beaker = obs[-1]
        target_color_str = f"RGB({target_beaker[0]}, {target_beaker[1]}, {target_beaker[2]})"
        target_description = f"Target beaker: {target_color_str}, amount: {target_beaker[3]}ml"

        text_repr = "\n".join(beaker_descriptions) + "\n" + target_description
        return text_repr
    
    def execute_instruction(self, instruction, render=True):
        # if render:
        #     self.render()
    
        ins_type, args = instruction
        # ins_str = '{}({})'.format(ins_type, ', '.join(map(str, args)))

        if ins_type == 'POUR':
            from_idx, to_idx, amt = args
            self.step((from_idx, to_idx, amt, 0, 0))
        elif ins_type == 'DONE':
            comp_idx, = args
            self.step((0, 0, 0, 1, comp_idx))

        if render:
            self.render()


    def load_state_from_file(self, file_path):
        with open(file_path, 'r') as file:
            self.beakers = []
            self.target_beaker = None

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
                    self.target_beaker = Paint(color, amount)

            self.set_num_beakers(len(self.beakers))

        # Reset other necessary environment variables
        self.step_count = 0
        self.prev_action = None

        # Return the initial observation
        return self._get_observation()
    


    def render(self, mode='human'):
        # Define window dimensions and colors
        screen_width = 700
        screen_height = 400
        background_color = (27, 29, 30)  # Black background
        foreground_color = (255, 255, 255)  # Replace with your desired foreground color

        # Initialize Pygame if it hasn't been initialized
        if not pygame.get_init():
            pygame.init()
            self.screen = pygame.display.set_mode((screen_width, screen_height))

        if self.screen == None:
            self.screen = pygame.display.set_mode((screen_width, screen_height))

        screen = self.screen

        # Create Pygame screen
        screen.fill(background_color)

        # Define beaker dimensions and positions
        beaker_width = 60  # 40 pixels
        beaker_height = 100  # Height of the beaker in pixels
        space_between_beakers = 20  # 10 pixels
        beaker_capacity = 200  # Max capacity of each beaker (in units, not pixels)
        total_width = beaker_width + space_between_beakers
        beaker_border_width = 1  # Thickness of beaker border
        y_pad = 120  # Padding from the bottom of the screen
        target_offset = 30  # Additional space before the target beaker

        # Centering the beakers
        total_beakers_width = (len(self.beakers) + 1) * total_width  # Including target beaker
        start_x = (screen_width - total_beakers_width) // 2
        
        # Font for labels
        font = pygame.font.SysFont(None, 24)
        amount_font = pygame.font.SysFont(None, 20)  # Font for displaying the amount
        rgb_font = pygame.font.SysFont(None, 17)  # Font for displaying the amount


        # Draw each beaker and label
        for i, beaker in enumerate(self.beakers):
            x_position = start_x + i * total_width

            # Fill beaker with paint
            paint_height = beaker.amount / beaker_capacity * beaker_height
            pygame.draw.rect(screen, beaker.color, (x_position, screen_height - y_pad - paint_height, beaker_width, paint_height))

            # Draw beaker border 
            pygame.draw.rect(screen, foreground_color, (x_position, screen_height - y_pad - beaker_height, beaker_width, beaker_height), beaker_border_width)

            # Add label
            label = font.render(str(i), True, foreground_color)
            screen.blit(label, (x_position + beaker_width // 2 - label.get_width() // 2, screen_height - y_pad + 5))
            
            rgb_code = f"({beaker.color[0]}, {beaker.color[1]}, {beaker.color[2]})"
            rgb_label = rgb_font.render(rgb_code, True, foreground_color)
            screen.blit(rgb_label, (x_position + beaker_width // 2 - rgb_label.get_width() // 2, screen_height - y_pad + 25))
            
            # Add label for beaker amount
            amount_label = amount_font.render(f"{beaker.amount:.0f}", True, foreground_color)
            screen.blit(amount_label, (x_position + beaker_width // 2 - amount_label.get_width() // 2, screen_height - y_pad - beaker_height - 20))

        # Draw the target beaker (further to the right)
        target_x_position = start_x + len(self.beakers) * total_width + target_offset
        paint_height = self.target_beaker.amount / beaker_capacity * beaker_height
        pygame.draw.rect(screen, self.target_beaker.color, (target_x_position, screen_height - y_pad - paint_height, beaker_width, paint_height))
        pygame.draw.rect(screen, foreground_color, (target_x_position, screen_height - y_pad - beaker_height, beaker_width, beaker_height), beaker_border_width)

        # Add text under the target beaker
        text = font.render('Target', True, foreground_color)
        screen.blit(text, (target_x_position + beaker_width // 2 - text.get_width() // 2, screen_height - y_pad + 5))
        
        # Add RGB code below the target beaker
        target_rgb_code = f"({self.target_beaker.color[0]}, {self.target_beaker.color[1]}, {self.target_beaker.color[2]})"
        target_rgb_label = rgb_font.render(target_rgb_code, True, foreground_color)
        screen.blit(target_rgb_label, (target_x_position + beaker_width // 2 - target_rgb_label.get_width() // 2, screen_height - y_pad + 25))
        
        # Add label for target beaker amount
        target_amount_label = amount_font.render(f"{self.target_beaker.amount:.0f}", True, foreground_color)
        screen.blit(target_amount_label, (target_x_position + beaker_width // 2 - target_amount_label.get_width() // 2, screen_height - y_pad - beaker_height - 20))


        # visualize previous action
        if self.prev_action is not None:
            from_idx = self.prev_action['from_idx']
            to_idx = self.prev_action['to_idx']
            actual_amt = self.prev_action['actual_amt']
            desired_amt = self.prev_action['desired_amt']
            is_done = self.prev_action['is_done']
            cmp_idx = self.prev_action['cmp_idx']

            if is_done:
                # Draw a red line under the beaker with index cmp_idx
                # Draw an arrow pointing upwards to the cmp_idx beaker
                cmp_x_position = start_x + cmp_idx * total_width + beaker_width / 2
                arrow_len = 25
                arrow_base = (cmp_x_position, screen_height - y_pad + 70)
                arrow_head = (cmp_x_position, screen_height - y_pad + 70 - arrow_len)
                pygame.draw.line(screen, foreground_color, arrow_base, arrow_head, 2)  # Green arrow
                pygame.draw.polygon(screen, foreground_color, [(arrow_head[0], arrow_head[1]), (arrow_head[0] - 5, arrow_head[1] + 10), (arrow_head[0] + 5, arrow_head[1] + 10)])

                done_font = pygame.font.SysFont(None, 48)  # Larger font for "Done" text
                done_text = done_font.render("Done", True, foreground_color)
                screen.blit(done_text, (screen_width // 2 - done_text.get_width() // 2, 20))  # Top center position
            else:
                # Calculate positions for the start and end of the arrow
                from_x = start_x + from_idx * total_width + beaker_width / 2
                to_x = start_x + to_idx * total_width + beaker_width / 2
                y = screen_height - y_pad - beaker_height / 2 - 100

                # Draw the arrow line
                pygame.draw.line(screen, foreground_color, (from_x, y), (to_x, y), 2)

                # Determine arrowhead orientation based on transfer direction
                if from_x < to_x:  # Transfer from left to right
                    arrowhead = [(to_x, y), (to_x - 10, y - 5), (to_x - 10, y + 5)]
                else:  # Transfer from right to left
                    arrowhead = [(to_x, y), (to_x + 10, y - 5), (to_x + 10, y + 5)]

                # Draw the arrowhead
                pygame.draw.polygon(screen, foreground_color, arrowhead)

                # Add label for the desired and actual transfer amounts
                desired_label = amount_font.render(f"Desired: {desired_amt:.0f}", True, foreground_color)
                actual_label = amount_font.render(f"Actual: {actual_amt:.0f}", True, foreground_color)
                mid_x = (from_x + to_x) / 2
                screen.blit(desired_label, (mid_x - desired_label.get_width() / 2, y - 40))
                screen.blit(actual_label, (mid_x - actual_label.get_width() / 2, y - 20))

        # Update the display
        pygame.display.flip()
        
        input('Press enter to continue...')

        # Keep the window open until clicked
        # wait_for_click = True
        # while wait_for_click:
        #     for event in pygame.event.get():
        #         if event.type == pygame.QUIT:
        #             wait_for_click = False
        #         elif event.type == pygame.MOUSEBUTTONDOWN:
        #             wait_for_click = False

if __name__ == '__main__':

    # Create the environment
    # env = ColorMixingEnv(
    #     beakers=[
    #         Paint((255, 0, 0), 100),
    #         Paint((0, 255, 0), 100),
    #         Paint((0, 0, 255), 100),
    #         Paint((0, 0, 0), 0) # empty beaker
    #     ],
    #     target_color=(128, 128, 0),  # Olive green
    #     target_amount=150
    # )
    env = ColorMixingEnv(
        num_beakers=4
    )
    
    env.load_state_from_file('tests/greenish.txt')
    print(env._get_observation())


    env.render()
    # print('initial score:', env.calculate_score(3))
    # # input()
    obs, reward, done, trunacated, _ = env.step((1, 3, 50, 0, 3))
    env.render()
    obs, reward, done, trunacated, _ = env.step((0, 3, 44, 0, 3))
    # print('step 1:', obs, reward, env.calculate_score(3))
    env.render()
    obs, reward, done, trunacated, _ = env.step((2, 3, 26, 0, 3))
    env.render()

    # obs, reward, done, trunacated, _ = env.step((1, 3, 75, 0, 3))

    # print('step 2:', obs, reward, env.calculate_score(3))
    # env.render()




