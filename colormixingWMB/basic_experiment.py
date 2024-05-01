from world_model import WorldModel, ColorMixingWMInit, WorldModelBuilder
from search import iterative_deepening_dfs
from env import ColorMixing
from data import get_color_mixing_level
from dist import * 
import random
import inspect
import logging
# from data import get_blocks_world_level
import uuid
import os
import pandas as pd
import json
    

def try_level(level_path, wmb : WorldModelBuilder, max_ep_num, max_ep_len, search_depth, level_pass_threshold=0.95):
    
    init_state, goal_state = get_color_mixing_level(level_path)
    # breakpoint()

    wmb.world_model.goal_state = goal_state
    
    env = ColorMixing() 

    achieved_total_success = False 
    num_attempts = 0 # number of episodes of experience needed 
    num_llm_queries = 0

    for _ in range(max_ep_num):

        curr_state = init_state
        num_attempts += 1
        
        episode_id = uuid.uuid4()

        achieved_goal = False
        trajectory = []

        # Stage 1: Interact with Environment (planning using world model)
        for i in range(max_ep_len):
            steps_left = max_ep_len - i
            action_seq, value = iterative_deepening_dfs(curr_state, wmb.world_model, min(search_depth, steps_left))
            # breakpoint()

            if len(action_seq) == 0:
                # random action
                action = random.choice(wmb.world_model.suggest_actions(curr_state))
            else:
                # print("actually have a move??", action_seq, value)
                action = action_seq[0]

            next_state = env.state_transition(curr_state, action)

            # print(f"curr_state={curr_state}") 
            # print(f"action={action}, action_seq={action_seq}, value={value}")
            # print(f"next_state={next_state}\n")

            trajectory.append({
                'state': curr_state,
                'action' : action,
                'pred_next_state' : wmb.world_model.state_transition(curr_state, action),
                'actual_next_state' : next_state
            })

            wmb.add_transition(curr_state, action, next_state, id=episode_id)
            curr_state = next_state
            
            
            if state_similarity(goal_state, curr_state) >= level_pass_threshold:
                achieved_goal = True
                print(trajectory)
                print("Achieved goal!")
                break
            # breakpoint()

        # if achieved_goal and pass_rate == 1:
        if achieved_goal:
            # achieved total sucess so we can stop
            achieved_total_success = True 
            break

        
        # Stage 2: Refine world model using experience
        prev_stf = wmb.stf_str

        num_llm_queries += wmb.refine_world_model()
        pass_rate = wmb.calculate_pass_rate_on_episode(episode_id)
        next_stf = wmb.stf_str
        
        if prev_stf == next_stf:
            print(f"no change to STF so failed level {level_path}")
            break
        
    # return statistics
    return achieved_total_success, num_attempts, num_llm_queries, trajectory

def load_frozen_test_cases():
    results_list = []
    path = "data/test_cases.json"
    f = open(path)
    data = json.load(f)
    
    pour_trans = []
    cnt = 0

    for action_type, trans_list in data.items():
        for trans in trans_list:
            input = {
                "state": set(trans["state"]),
                "action": trans["action"],
                "episode_id": cnt
            } 
            next_state = set(trans["next_state"])
            
            pour_trans.append((input, next_state))
            cnt += 1

    return pour_trans


def main():
    

    # cm_env = ColorMixing()
    # state = {
    #     "contains 0 255 255 255 100",
    #     "contains 1 0 0 0 100",
    # }
    # action = "pour 0 1 100"
    # next_state = cm_env.state_transition(state, action)
    # print(next_state)
    
    # cm_wm = ColorMixingWMInit()
    # print(cm_wm.suggest_actions(state))
    # print(state_similarity(state, next_state))
    
    datadir = "data/level1"
    pour_trans = load_frozen_test_cases()

    max_ep_len = 12 
    search_depth = 3
    max_ep_num = 5
    max_refine_iters=3
    
    num_level_per_dir = 10 

    # new hyper parameters (need to link) 
    level_pass_threshold = None
    tc_sim_refinement_threshold = None

    wmb = WorldModelBuilder(world_model=ColorMixingWMInit(), max_refine_iters=max_refine_iters)
    data_columns = ["filename", "passed", "num_attempts", "num_llm_queries", "last_trajectory", "experience_sim_score", "fixed_tc_sim_score"]
    data_df = pd.DataFrame(columns=data_columns)

    for filename in os.listdir(datadir):
        full_path = os.path.join(datadir, filename)

        res = try_level(full_path, wmb, max_ep_num, max_ep_len, search_depth)
        pass_rate = wmb.calculate_avg_sim_on_tc(wmb.get_test_results())
        
        fixed_tc_pass_rate = wmb.calculate_avg_sim_on_tc(wmb.get_test_results(pour_trans))

        data_point = (filename, *res, pass_rate, fixed_tc_pass_rate) 
        print(data_point)

        new_data_df = pd.DataFrame([data_point], columns=data_columns)
        data_df = pd.concat([data_df, new_data_df], ignore_index=True)
        
        data_df.to_csv("results/basic_experiment.csv", index=False)
        


def test():   
    # curr_state, goal_state = get_color_mixing_level('data/level1/beige.json')
    # search_depth = 3
    # wmb = WorldModelBuilder(world_model=ColorMixingWMInit(), max_refine_iters=10)
    # wmb.world_model.goal_state = goal_state
    # action_seq, value = iterative_deepening_dfs(curr_state, wmb.world_model, search_depth)
    # print(action_seq, value)
    
    # cm = ColorMixing()
    
    # for action in action_seq:
    #     next_state = cm.state_transition(curr_state, action)    
    #     print(action, sorted(list(next_state)), state_similarity(goal_state, next_state))
    #     curr_state = next_state
    
    pred = ["contains 1 243 0 5 105", "contains 2 0 204 0 125", "contains 3 0 0 178 101", "contains 4 255 255 255 100", "contains 5 0 0 102 69", "contains 6 0 0 0 1"]
    actual = ["contains 1 243 0 5 105", "contains 2 0 204 0 125", "contains 3 0 0 178 101", "contains 4 255 255 255 100", "contains 5 0 0 102 68", "contains 6 255 255 255 1"]

    for i, (b1, b2) in enumerate(zip(pred, actual)):
        print(i, beaker_similarity(b1, b2))

    # print(state_similarity(pred, actual))
    





if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # test()
    # yes = load_frozen_test_cases()
    # breakpoint()
    main()