import abc
from dataclasses import dataclass
from prompts import wmb_template 
from str_parse import extract_section

from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain

import types
import inspect
import logging

import random


from textwrap import dedent

class WorldModel():
    # def __init__(self):
    #     self.objects = None
    #     self.predicates = None
    #     self.actions = None

    @abc.abstractmethod
    def state_value(self, state):
        """
        returns scalar score value for given state (can also be written by LLM?)
        """
        pass

    @abc.abstractmethod
    def suggest_actions(self, state):
        """
        returns a list of actions 
        """
        pass

    @abc.abstractmethod 
    def state_transition(self, state, action):
        """
        returns new state 
        """
        pass

# initial blocks world model with empty state transition function, but 
# implemented state_value and suggest_actions functions
class BlocksWorldModelInit(WorldModel):
    def __init__(self, goal_state=None):
        self.goal_state = goal_state

    def state_value(self, state):
        sat_pred_cnt = sum(predicate in state for predicate in self.goal_state)
        score = sat_pred_cnt / len(self.goal_state)
        return score
        
    def suggest_actions(self, state):
        """
        Finds all valid actions based on the current state.
        """
        valid_actions = []

        # Determine if the arm is empty
        arm_empty = "handempty" in state
        holding_something = not arm_empty

        clear_blocks = {predicate.split()[1] for predicate in state if predicate.startswith("clear")}
        on_table_blocks = {predicate.split()[1] for predicate in state if predicate.startswith("ontable")}
        holding_block = next((predicate.split()[1] for predicate in state if predicate.startswith("holding")), None)

        if arm_empty:
            # If the arm is empty, we can pickup any clear block that is on the table
            for block in clear_blocks & on_table_blocks:
                valid_actions.append(f"pick-up {block}")

        if holding_something:
            # If holding a block, can put it down on the table or stack it on another clear block
            valid_actions.append(f"put-down {holding_block}")
            for block in clear_blocks:
                if block != holding_block:  # Avoid stacking a block on itself
                    valid_actions.append((f"stack {holding_block} {block}"))

        # Check for blocks that can be unstacked (on top of another block and clear)
        for predicate in state:
            if predicate.split()[0] == "on" and predicate.split()[1] in clear_blocks:
                _, ob, underob = predicate.split()
                # Ensure underob is clear and arm is empty before adding unstack action
                # breakpoint()
                if "clear " + ob in state and arm_empty:
                    valid_actions.append(f"unstack {ob} {underob}")

        return valid_actions
    
    def state_transition(self, state, action):
        """
        Applies an action to the current state and returns the new state.

        :param state: A set of predicates (strings) representing the current state.
        :param action: A string representing the action to be applied.
        :return: The new state as a set of predicates.
        """

        # Split action into words to extract action type and parameters
        words = action.split()
        action_type = words[0]
        params = words[1:]
        next_state = set(state)
        # implement the rest!
        return next_state

    # def state_transition(self, state, action):
    #     words = action.split()
    #     action_type = words[0]
    #     params = words[1:]
    #     next_state = set(state)

    #     if action_type == "pick-up":
    #         next_state.discard("handempty")
    #         next_state.add("holding " + params[0])
    #         next_state.discard("ontable " + params[0])
    #         next_state.discard("clear " + params[0])

    #     elif action_type == "put-down":
    #         next_state.add("handempty")
    #         next_state.add("ontable " + params[0])
    #         next_state.add("clear " + params[0])
    #         next_state.discard("holding " + params[0])

    #     elif action_type == "stack":
    #         next_state.add("clear " + params[0])
    #         next_state.discard("ontable " + params[0])
    #         next_state.discard("clear " + params[1])
    #         next_state.add("on " + params[0] + " " + params[1])

    #     elif action_type == "unstack":
    #         next_state.add("clear " + params[1])
    #         next_state.add("ontable " + params[0])
    #         next_state.discard("on " + params[0] + " " + params[1])

    #     return next_state




# LLM which acts as the world model builder
class WorldModelBuilder:
    def __init__(self, world_model : WorldModel, max_refine_iters=3):
        self.transitions = [] # transition bank
        self.world_model = world_model

        # string representation state transition function       
        self.stf_str = inspect.getsource(self.world_model.state_transition) 

        llm = ChatOpenAI(
            temperature=0,
            model_name="gpt-3.5-turbo-1106"
            # model_name='gpt-4-1106-preview'
        )
        self.llm = LLMChain(llm=llm, prompt=wmb_template)
        
        self.max_refine_iters = max_refine_iters 
        
    
    def refine_world_model(self, init=False) -> int:
        # test the existing world model
        results = self.get_test_results()
        results_str = self.filter_and_format_results(results)
        pass_rate = self.calculate_pass_rate(results)

        best_stf_str = dedent(self.stf_str)
        best_pass_rate = pass_rate
        
        num_llm_queries = 0
        
        logging.info(f"Initial results:")
        logging.info(f" Results_str: {results_str}")
        logging.info(f" Pass_rate: {pass_rate}")
        
        for _ in range(self.max_refine_iters):
            # we can stop if all test cases pass:
            if pass_rate == 1:
                break
            
            logging.info(f"Pass rate not high enough so querying WMB...")
            num_llm_queries += 1
            wmb_response = self.llm.run(
                prev_stf = self.stf_str,
                tester_res = results_str
            )
            logging.info(f"wmb_response: {wmb_response}")

            # ! Need to deal with potential syntax errors here 
            # ! check if test cases run? regenerate until all test cases pass?
            new_gen_stf = dedent(extract_section(wmb_response, "CODE"))
            # print("Extracted code: ", new_gen_stf )
            # self.stf_str = new_gen_stf 
            # breakpoint()
            
            # modify the stf method in the world model 
            namespace = {}
            exec(new_gen_stf, globals(), namespace)  # compile the string into a function
            stf = namespace["state_transition"]
            self.world_model.state_transition = types.MethodType(stf, self.world_model)
            
            # test the new world model
            results = self.get_test_results()
            # results_str = self.filter_and_format_results(results, num_fail=1)
            pass_rate = self.calculate_pass_rate(results)
            
            # logging.info(f" new_results: {results_str}")
            logging.info(f" new pass_rate after refining: {pass_rate}")

            if pass_rate >= best_pass_rate or init:
                best_pass_rate = pass_rate
                best_stf_str = new_gen_stf

            # revert the world back to the best performing stf
            # breakpoint()
            namespace = {}
            exec(best_stf_str, globals(), namespace)
            stf = namespace["state_transition"]
            self.world_model.state_transition = types.MethodType(stf, self.world_model)
            self.stf_str=best_stf_str
            
            if init:
                break
        
        # return statistics
        return num_llm_queries 


    def add_transition(self, state, action, next_state, id=None):
        input = {
            "state": state,
            "action": action,
            "episode_id": id
        }
        self.transitions.append((input, next_state))
        
    def get_test_results(self, transitions=None):
        # run the test cases (self.transitions) for the current state transition function
        results_list = []
        results_str = ""
        if transitions == None:
            transitions = self.transitions

        for i, (input, next_state) in enumerate(transitions):
            state, action = input["state"], input["action"]
            pred_next_state = self.world_model.state_transition(state, action)

            results_list.append({
                "state" : state,
                "action" : action,
                "expected_out" : next_state,
                "actual_out" : pred_next_state,
                "id" : i, # test case id
                "episode_id" : input["episode_id"]
            })

        return results_list

    def filter_and_format_results(self, results : list[dict], num_fail=1):
        def sorted_set_str(s):
            sorted_list = sorted(list(s))
            sorted_set_str = "{" + ", ".join(str(e) for e in sorted_list) + "}"
            return sorted_set_str

        # can also only show a max of k test cases which fail
        results_str = ""
        
        cnt = 0
        
        random.shuffle(results)
        
        for tc in results:
            id = tc["id"]
            state = tc["state"]
            action = tc["action"]
            next_state, pred_next_state = tc["expected_out"], tc["actual_out"]

            result = "Pass" if pred_next_state == next_state else "Fail"
            
            # only keep test cases which fail
            if result == "Pass":
                continue
            

            test_case_str = dedent(f"""
            Test case {id}:
                Input:
                    State: {sorted_set_str(state)}
                    Action: {action}
                Predicted next_state: {sorted_set_str(pred_next_state)}
                Actual next_state: {sorted_set_str(next_state)}
                Result: {result}
            """)

            results_str += test_case_str

            cnt += 1
            if cnt >= num_fail:
                break
        
        return results_str
    
    def calculate_pass_rate(self, results : list[dict]):
        if len(results) == 0:
            return 0

        pass_cnt = 0
        for tc in results:
            if tc["expected_out"] == tc["actual_out"]:
                pass_cnt += 1
        
        return pass_cnt / len(results)
    
    def calculate_pass_rate_on_episode(self, episode_id):
        all_results = self.get_test_results()
        episode_results = []
        for res in all_results:
            if res["episode_id"] == episode_id:
                episode_results.append(res)
        return self.calculate_pass_rate(episode_results)




def main():
    wmb = WorldModelBuilder()
    wmb.refine_world_model()

if __name__ == "__main__":
    main()

