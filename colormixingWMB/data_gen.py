import json
import random
import os
from pathlib import Path

def generate_color_mixing_data(num_files=100, output_dir="data/generated"):
    """
    Generate random color mixing JSON files with fixed initial state and random goal state.
    
    Args:
        num_files: Number of JSON files to generate
        output_dir: Directory to save the generated files
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    initial_state = [
        "contains 1 255 0 0 100",    # Red
        "contains 2 0 255 0 100",    # Green
        "contains 3 0 0 255 100",    # Blue
        "contains 4 255 255 255 100", # White
        "contains 5 0 0 0 100",      # Black
        "contains 6 0 0 0 0"         # Empty
    ]
    
    # Parameters for goal state generation
    min_amount = 50
    max_amount = 150
    
    for i in range(num_files):
        # Generate random RGB values for goal state
        r = random.randint(0, 255)
        g = random.randint(0, 255)
        b = random.randint(0, 255)
        amount = random.randint(min_amount, max_amount)
        
        # Randomly select a target beaker (1-6)
        target_beaker = random.randint(1, 6)
        
        # Create goal state with random target beaker
        goal_state = [f"contains {target_beaker} {r} {g} {b} {amount}"]
        
        # Create data dictionary
        data = {
            "state": initial_state,
            "goal_state": goal_state
        }
        
        # Save to JSON file
        filename = f"color_mix_{i+1}.json"
        with open(os.path.join(output_dir, filename), 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"Generated {filename}")

if __name__ == "__main__":
    generate_color_mixing_data() 