from world_model import WorldModel, BlocksWorldModelInit, WorldModelBuilder
from search import iterative_deepening_dfs
from env import BlocksWorld
import random
import inspect
from data import get_blocks_world_level
import uuid



# def main():
#     max_ep_len = 5 
#     search_depth = 7
#     max_ep_num = 2
    
#     # init_state, goal_state = get_blocks_world_level("data_parsed/step_6/instance-11.pddl")
#     init_state, goal_state = get_blocks_world_level("data_parsed/step_2/instance-5.pddl")
    
#     wmb = WorldModelBuilder(world_model=BlocksWorldModelInit(goal_state))
#     env = BlocksWorld() 
    
#     # generate initial world model
#     # wmb.refine_world_model(num_iter=1)

#     for _ in range(max_ep_num):

#         curr_state = init_state
#         achieved_goal = False
        
#         episode_id = uuid.uuid4()

#         # Stage 1: Interact with Environment (planning using world model)
#         for _ in range(max_ep_len):
#             action_seq, value = iterative_deepening_dfs(curr_state, wmb.world_model, search_depth)
#             # breakpoint()

#             if len(action_seq) == 0:
#                 # random action
#                 action = random.choice(wmb.world_model.suggest_actions(curr_state))
#             else:
#                 action = action_seq[0]

#             next_state = env.state_transition(curr_state, action)

#             print(f"curr_state={curr_state}") 
#             print(f"action={action}, action_seq={action_seq}, value={value}")
#             print(f"next_state={next_state}\n")

#             wmb.add_transition(curr_state, action, next_state, id=episode_id)
#             curr_state = next_state
            
#             if goal_state <= curr_state:
#                 achieved_goal = True
#                 print("Achieved goal!")
#                 break
        
#         # if achieved_goal:
#         #     # skip refining world model
#         #     break

#         # Stage 2: Refine world model using experience
#         wmb.refine_world_model()


# if __name__ == "__main__":
#     main()


    # current_state = {
    #     "on-table a", "on-table b", "clear a", "clear b", "on c a", "clear c", "arm-empty"
    # }
    # current_state = {
    # "on-table a", "on-table b", "on c a", "clear b", "clear c", "arm-empty"
    # }
    
    # print(wm.suggest_actions(current_state))
    # action = "pick-up c"
    # current_state = wm.state_transition(current_state, "pick-up c")
    # print(current_state)
    # print(wm.suggest_actions(current_state))
    # valid_actions = wm.suggest_actions(current_state)
    # for action, params in valid_actions:
    #     print(f"Action: {action}, Parameters: {params}")

    # print(iterative_deepening_dfs(current_state, wm, 4))
    


    # current_state = wm.state_transition(current_state, 'pickup', ['c'])
    # print(current_state)
    # print(wm.suggest_actions(current_state))
    # current_state = wm.state_transition(current_state, 'stack', ['c', 'a'])
    # print(current_state)
    # print(wm.state_value(current_state))



