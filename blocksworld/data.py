import json

def get_blocks_world_level(path):
    f = open(path)
    data = json.load(f)
    return set(data['init_state']), set(data['goal_state'])

