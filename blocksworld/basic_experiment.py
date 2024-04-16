from world_model import WorldModel, BlocksWorldModelInit, WorldModelBuilder
from search import iterative_deepening_dfs
from env import BlocksWorld
import random
import inspect
import logging
from data import get_blocks_world_level
import uuid
import os
import pandas as pd
import json
    

def try_level(level_path, wmb : WorldModelBuilder, max_ep_num, max_ep_len, search_depth):
    
    init_state, goal_state = get_blocks_world_level(level_path)

    # set the goal state of the world model
    wmb.world_model.goal_state = goal_state
    
    env = BlocksWorld() 

    achieved_total_success = False 
    num_attempts = 0 # number of episodes of experience needed 
    num_llm_queries = 0

    for _ in range(max_ep_num):

        curr_state = init_state
        num_attempts += 1
        
        episode_id = uuid.uuid4()

        achieved_goal = False
        # Stage 1: Interact with Environment (planning using world model)
        for _ in range(max_ep_len):
            action_seq, value = iterative_deepening_dfs(curr_state, wmb.world_model, search_depth)
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

            wmb.add_transition(curr_state, action, next_state, id=episode_id)
            curr_state = next_state
            
            if goal_state <= curr_state:
                achieved_goal = True
                print("Achieved goal!")
                break

        # if achieved_goal and pass_rate == 1:
        if achieved_goal:
            # achieved total sucess so we can stop
            achieved_total_success = True 
            break
        
        # Stage 2: Refine world model using experience
        num_llm_queries += wmb.refine_world_model()
        pass_rate = wmb.calculate_pass_rate_on_episode(episode_id)

    
    # return statistics
    return achieved_total_success, num_attempts, num_llm_queries

def load_frozen_test_cases():
    results_list = []
    path = "data_parsed/test_cases.json"
    f = open(path)
    data = json.load(f)
    
    stack_trans = [] 
    unstack_trans = [] 
    pick_up_trans = [] 
    put_down_trans = [] 
    
    cnt = 0

    for action_type, trans_list in data.items():
        for trans in trans_list:
            input = {
                "state": set(trans["state"]),
                "action": trans["action"],
                "episode_id": cnt
            } 
            next_state = set(trans["next_state"])
            
            if action_type == "stack":
                stack_trans.append((input, next_state))
            elif action_type == "unstack":
                unstack_trans.append((input, next_state))
            elif action_type == "pick-up":
                pick_up_trans.append((input, next_state))
            elif action_type == "put-down":
                put_down_trans.append((input, next_state))
            else:
                assert False
            # breakpoint()
            cnt += 1

    return stack_trans, unstack_trans, pick_up_trans, put_down_trans


def main():
    # datadir = "data_parsed/step_2"
    datadirs = [
        "data_parsed/step_2",
        "data_parsed/step_4",
        "data_parsed/step_6",
    ]

    stack_trans, unstack_trans, pick_up_trans, put_down_trans = load_frozen_test_cases()
    # breakpoint()

    # hyperparameters
    max_ep_len = 7 
    search_depth = 7
    max_ep_num = 5
    max_refine_iters=3
    
    num_level_per_dir = 10 
    
    wmb = WorldModelBuilder(world_model=BlocksWorldModelInit(), max_refine_iters=max_refine_iters)

    # generate initial world model
    # wmb.refine_world_model(init=True)

    data_columns = ["filename", "passed", "num_attempts", "num_llm_queries", "experience_PR", "stack_PR", "unstack_PR", "pick_up_PR", "put_down_PR"]
    data_df = pd.DataFrame(columns=data_columns)

    for datadir in datadirs:
        cnt = 0
        for filename in os.listdir(datadir):
            full_path = os.path.join(datadir, filename)

            res = try_level(full_path, wmb, max_ep_num, max_ep_len, search_depth)

            pass_rate = wmb.calculate_pass_rate(wmb.get_test_results())
            stack_pr = wmb.calculate_pass_rate(wmb.get_test_results(stack_trans))
            unstack_pr = wmb.calculate_pass_rate(wmb.get_test_results(unstack_trans))
            pick_up_pr = wmb.calculate_pass_rate(wmb.get_test_results(pick_up_trans))
            put_down_pr = wmb.calculate_pass_rate(wmb.get_test_results(put_down_trans))

            data_point = (filename, *res, pass_rate, stack_pr, unstack_pr, pick_up_pr, put_down_pr) 
            print(data_point)

            new_data_df = pd.DataFrame([data_point], columns=data_columns)
            data_df = pd.concat([data_df, new_data_df], ignore_index=True)
            
            data_df.to_csv("results/basic_experiment.csv", index=False)
            
            cnt += 1 
            if cnt >= num_level_per_dir:
                break

            # breakpoint()

    # print("Solved level in {}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()