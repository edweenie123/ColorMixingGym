from env import *
import random
import numpy as np


def get_env_text_repr(obs):
    """
    Receives observation from environment (env state) and converts it
    to a string representation.
    Each beaker is represented by a line with 'B' or 'T' followed by its color in RGB and amount.
    'B' is for normal beakers and 'T' for the target beaker.
    """
    beaker_descriptions = []
    num_beakers = len(obs) - 1  # Excluding the target beaker

    # Process each beaker
    for idx in range(num_beakers):
        beaker = obs[idx]
        beaker_descriptions.append(f"B {beaker[0]} {beaker[1]} {beaker[2]} {beaker[3]}")

    # Process the target beaker
    target_beaker = obs[-1]
    target_description = f"T {target_beaker[0]} {target_beaker[1]} {target_beaker[2]} {target_beaker[3]}"

    text_repr = "\n".join(beaker_descriptions) + "\n" + target_description
    return text_repr

def generate_level2(num):
    env = ColorMixingEnv(num_beakers=5) 
    
    for i in range(num):
        primary = [
                (255, 0, 0),  # cyan
                (0, 255, 0),  # megenta
                (0, 0, 255)   # yellow
        ]

        beakers = []

        # apply noise to each primary color 
        for col in primary:
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

        rand_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        rand_amount = random.randint(40, 120) # set suitable min and max amounts

        random.shuffle(beakers)

        env.beakers = beakers

        env.target_beaker = Paint(rand_color, rand_amount)
        # set the rest to be empty beakers
        for _ in range(self.num_beakers - 4):
            beakers.append(Paint((0, 0, 0), 0))



if __name__ == '__main__':

