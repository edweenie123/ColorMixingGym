from world_model import WorldModel 


def depth_limited_dfs(init_state, world_model, limit):
    """
    Perform depth-limited DFS in a graph to find the action sequence leading to the state with the highest value.
    
    :param init_state: The initial state from which to start the DFS.
    :param world_model: An object representing the world, with actions, state_value, and state_transition defined.
    :param limit: The depth limit for the DFS search.
    :return: The sequence of actions leading to the state with the highest value.
    """
    def dfs_recursive(state, depth, path, max_value, best_path):
        if depth == limit:
            current_value = world_model.state_value(state)
            if current_value > max_value[0]:
                max_value[0] = current_value
                best_path[:] = path[:]
            return
        
        for action in world_model.suggest_actions(state):
            # breakpoint()
            new_state = world_model.state_transition(state, action)
            dfs_recursive(new_state, depth + 1, path + [action], max_value, best_path)
    
    max_value = [-float('inf')]
    best_path = []
    dfs_recursive(init_state, 0, [], max_value, best_path)
    
    return best_path, max_value

def iterative_deepening_dfs(init_state, world_model, limit):
    """
    Perform Iterative Deepening DFS to find the action sequence leading to the state with the highest value.
    
    :param init_state: The initial state from which to start the DFS.
    :param world_model: An object representing the world, with suggest_actions, state_value, and state_transition defined.
    :return: The sequence of actions leading to the state with the highest value, along with its value.
    """
    depth = 0
    best_path_global = []
    max_value_global = [-float('inf')]

    # Keep iterating, increasing the depth limit with each iteration
    while True:
        best_path, max_value = depth_limited_dfs(init_state, world_model, depth)
        
        # Check if the current iteration found a better solution
        if max_value[0] > max_value_global[0]:
            max_value_global = max_value
            best_path_global = best_path
        
        if depth == limit:  
            break

        depth += 1

    return best_path_global, max_value_global[0]