import pygame
from typing import List, Dict, Tuple
from dist import state_similarity
import pandas as pd 
import json

class Beaker:
    def __init__(self, color: Tuple[int, int, int], amount: float):
        self.color = color
        self.amount = amount

def parse_beaker(beaker_string):
    _, idx, r, g, b, amt = beaker_string.split()
    return int(idx), Beaker((int(r), int(g), int(b)), float(amt))


def visualize_trajectory(trajectory : List[Dict], goal_state_str):
    pygame.init()

    scale = 1.5 

    # Define window dimensions and colors
    screen_width = int(700 * scale)
    screen_height = int(400 * scale)
    background_color = (27, 29, 30)  # Black background 
    foreground_color = (255, 255, 255)  # Replace with your desired foreground color

    screen = pygame.display.set_mode((screen_width, screen_height))

    # Create Pygame screen
    screen.fill(background_color)

    # Define beaker dimensions and positions
    beaker_width = int(60 * scale) # 40 pixels
    beaker_height = int(100 * scale) # Height of the beaker in pixels
    space_between_beakers = int(20 * scale)  # 10 pixels
    beaker_capacity = 200  # Max capacity of each beaker (in units, not pixels)
    total_width = beaker_width + space_between_beakers
    beaker_border_width = int(1  * scale) # Thickness of beaker border
    y_pad = int(120 * scale) # Padding from the bottom of the screen
    target_offset = int(30 * scale) # Additional space before the target beaker
    prev_action_arrow_padding = int(100 * scale)
    
    num_beakers = 6 # ! keep this hardcoded for now...

    # Centering the beakers
    total_beakers_width = (num_beakers + 1) * total_width  # Including target beaker
    start_x = (screen_width - total_beakers_width) // 2
    
    # Font for labels
    font = pygame.font.SysFont(None, 28)
    amount_font = pygame.font.SysFont(None, 24)  # Font for displaying the amount
    rgb_font = pygame.font.SysFont(None, 21)  # Font for displaying the amount

    # ! assuming there is only one predicate in goal state for now  
    goal_beaker_idx, goal_beaker = parse_beaker(goal_state_str[0])
    
    prev_action = None
    
    last_state = trajectory[-1]["actual_next_state"]
    trajectory.append({
        "state" : last_state
    })

    
    for j, t in enumerate(trajectory):
        
        # please extract the correct informatino from t to get <self.beakers> 
        # also extract self.target_beaker from goal state
        # you can assume goal_state is just a single contains predicate
        # breakpoint()
        
        current_state_beakers = [parse_beaker(beaker_str) for beaker_str in t['state']]
        current_state_beakers.sort(key=lambda x: x[0])
        # Extract beaker objects for drawing
        beakers = [beaker for _, beaker in current_state_beakers]
        


        # Draw each beaker and label
        for i, beaker in enumerate(beakers):
            x_position = start_x + i * total_width

            # Fill beaker with paint
            paint_height = beaker.amount / beaker_capacity * beaker_height
            pygame.draw.rect(screen, beaker.color, (x_position, screen_height - y_pad - paint_height, beaker_width, paint_height))

            # Draw beaker border 
            pygame.draw.rect(screen, foreground_color, (x_position, screen_height - y_pad - beaker_height, beaker_width, beaker_height), beaker_border_width)

            # Add label
            label = font.render(str(i + 1), True, foreground_color)
            screen.blit(label, (x_position + beaker_width // 2 - label.get_width() // 2, screen_height - y_pad + 5))
            
            rgb_code = f"({beaker.color[0]}, {beaker.color[1]}, {beaker.color[2]})"
            rgb_label = rgb_font.render(rgb_code, True, foreground_color)
            screen.blit(rgb_label, (x_position + beaker_width // 2 - rgb_label.get_width() // 2, screen_height - y_pad + 25))
            
            # Add label for beaker amount
            amount_label = amount_font.render(f"{beaker.amount:.0f}", True, foreground_color)
            screen.blit(amount_label, (x_position + beaker_width // 2 - amount_label.get_width() // 2, screen_height - y_pad - beaker_height - 20))

            # Draw the target beaker (further to the right)
            target_x_position = start_x + len(beakers) * total_width + target_offset
            paint_height = goal_beaker.amount / beaker_capacity * beaker_height
            pygame.draw.rect(screen, goal_beaker.color, (target_x_position, screen_height - y_pad - paint_height, beaker_width, paint_height))
            pygame.draw.rect(screen, foreground_color, (target_x_position, screen_height - y_pad - beaker_height, beaker_width, beaker_height), beaker_border_width)

        # Add text under the target beaker
        text = font.render(f"Target ({goal_beaker_idx})", True, foreground_color)
        screen.blit(text, (target_x_position + beaker_width // 2 - text.get_width() // 2, screen_height - y_pad + 5))
        
        # Add RGB code below the target beaker
        target_rgb_code = f"({goal_beaker.color[0]}, {goal_beaker.color[1]}, {goal_beaker.color[2]})"
        target_rgb_label = rgb_font.render(target_rgb_code, True, foreground_color)
        screen.blit(target_rgb_label, (target_x_position + beaker_width // 2 - target_rgb_label.get_width() // 2, screen_height - y_pad + 25))
        
        # Add label for target beaker amount
        target_amount_label = amount_font.render(f"{goal_beaker.amount:.0f}", True, foreground_color)
        screen.blit(target_amount_label, (target_x_position + beaker_width // 2 - target_amount_label.get_width() // 2, screen_height - y_pad - beaker_height - 20))
        
        # visualize previous action
        if prev_action is not None:
            from_idx = prev_action[0] - 1
            to_idx = prev_action[1] - 1
            actual_amt = prev_action[2]
            # breakpoint()

            # Calculate positions for the start and end of the arrow
            from_x = start_x + from_idx * total_width + beaker_width / 2
            to_x = start_x + to_idx * total_width + beaker_width / 2
            y = screen_height - y_pad - beaker_height / 2 - prev_action_arrow_padding

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
            # desired_label = amount_font.render(f"Desired: {desired_amt:.0f}", True, foreground_color)
            actual_label = amount_font.render(f"Actual: {actual_amt:.0f}", True, foreground_color)
            mid_x = (from_x + to_x) / 2
            # screen.blit(desired_label, (mid_x - desired_label.get_width() / 2, y - 40))
            screen.blit(actual_label, (mid_x - actual_label.get_width() / 2, y - 20))
            
        if "action" in t:    
            _, src_idx, tar_idx, amt  = t["action"].split(" ")
            prev_action = (int(src_idx), int(tar_idx), int(amt))
        else:
            # Draw a red line under the beaker with index cmp_idx
            # Draw an arrow pointing upwards to the cmp_idx beaker
            # cmp_x_position = start_x + cmp_idx * total_width + beaker_width / 2
            # arrow_len = 25
            # arrow_base = (cmp_x_position, screen_height - y_pad + 70)
            # arrow_head = (cmp_x_position, screen_height - y_pad + 70 - arrow_len)
            # pygame.draw.line(screen, foreground_color, arrow_base, arrow_head, 2)  # Green arrow
            # pygame.draw.polygon(screen, foreground_color, [(arrow_head[0], arrow_head[1]), (arrow_head[0] - 5, arrow_head[1] + 10), (arrow_head[0] + 5, arrow_head[1] + 10)])

            done_font = pygame.font.SysFont(None, 48)  # Larger font for "Done" text
            done_text = done_font.render("Done", True, foreground_color)
            screen.blit(done_text, (screen_width // 2 - done_text.get_width() // 2, 20))  # Top center position

        # Update the display
        pygame.display.flip()
        screen.fill(background_color)

        print(f"Step {j}: ")
        print(f"Sim score: {state_similarity(goal_state_str, t['state'])}")
        input('Press enter to continue...\n')


def main():
    df = pd.read_csv("results/basic_experiment5.csv")
    dir = "data/level1"
    filename = "magenta.json"
    f = open(f"{dir}/{filename}", "r")
    data = df[df["filename"] == filename]
    trajectory = eval(data["last_trajectory"].item())
    f_data = json.load(f)
    
    # init_state = f_data["state"]
    goal_state = f_data["goal_state"]
    

    # breakpoint()
    # yes =     

    visualize_trajectory(trajectory, goal_state)

if __name__  == "__main__":
    main()
