import json

def get_color_mixing_level(path):
    f = open(path)
    data = json.load(f)
    return set(data['state']), set(data['goal_state'])

