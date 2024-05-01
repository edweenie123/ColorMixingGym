import abc
from dataclasses import dataclass
from prompts import wmb_template 
from str_parse import extract_section

from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from env import SubtractiveModel
from dist import *

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

# initial color mxiing model with empty state transition function, but 
# implemented state_value and suggest_actions functions
class ColorMixingWMInit(WorldModel):
    def __init__(self, goal_state=None):
        self.goal_state = goal_state

    def state_value(self, state):
        # Convert goal state to dictionary for easy lookup
        # goal_state_dict = {int(beaker.split()[1]): beaker for beaker in self.goal_state}

        # # Initialize total similarity
        # total_similarity = 0

        # # Calculate similarity for each beaker in the current state
        # for beaker in state:
        #     idx = int(beaker.split()[1])  # Extract the index of the beaker
        #     if idx in goal_state_dict:
        #         total_similarity += beaker_similarity(beaker, goal_state_dict[idx])

        # # Compute the average similarity
        # average_similarity = total_similarity / len(state) if state else 0
        return state_similarity(state, self.goal_state)

        
    def suggest_actions(self, state, goal_pruning=True):
        """
        Suggest promising actions from the given state
        """
        actions = []
        beakers = []
    
        # Parse state to create a list of tuples (idx, amount)
        for predicate in state:
            parts = predicate.split()
            idx = int(parts[1])
            amount = int(parts[5])
            beakers.append((idx, amount))


        # Generate actions for each pair of beakers
        for src in beakers:
            for tgt in beakers:
                if src[0] != tgt[0]:  # Ensure we do not pour into the same beaker
                    # Define possible pour amounts
                    pour_amounts = [5, 10, 25, 50, 100]
                    
                    # ! try high pour amounts first
                    pour_amounts.reverse() 

                    if goal_pruning:
                        if src[0]:
                            e = next(iter(self.goal_state))
                            goal_idx = int(e.split()[1])
                            if src[0] != goal_idx and tgt[0] != goal_idx:
                                continue


                    for amount in pour_amounts:
                        if src[1] >= amount:  # Check if the source has enough paint to pour
                            action = f"pour {src[0]} {tgt[0]} {amount}"
                            actions.append(action)
        
        random.shuffle(actions)
        return actions
    
    def state_transition(self, state, action):
        """
        Applies an action to the current state and returns the new state.

        :param state: A set of predicates (strings) representing the current state.
        :param action: A string representing the action to be applied.
        :return: The new state as a set of predicates.
        """
        
        def find_element(my_set, condition):
            for element in my_set:
                if condition(element):
                    return element
            return None  

        # Split action into words to extract action type and parameters
        words = action.split()
        action_type = words[0]
        params = words[1:]

        # Copy the current state to avoid mutating the original
        new_state = set(state)
        
        if action_type == "pour":
            src_idx, tgt_idx, amt = [int(x) for x in params]
            src_contains = find_element(state, lambda x: x.split()[1] == str(src_idx))
            tgt_contains = find_element(state, lambda x: x.split()[1] == str(tgt_idx))

            src_r, src_g, src_b, src_amt = [int(x) for x in src_contains.split()[2:]]
            tgt_r, tgt_g, tgt_b, tgt_amt = [int(x) for x in tgt_contains.split()[2:]]
            
            # implement the rest

        return new_state
    # def state_transition(self, state, action):
    #     """
    #     Applies an action to the current state and returns the new state.

    #     :param state: A set of predicates (strings) representing the current state.
    #     :param action: A string representing the action to be applied.
    #     :return: The new state as a set of predicates.
    #     """

    #     def find_element(my_set, condition):
    #         for element in my_set:
    #             if condition(element):
    #                 return element
    #         return None  

    #     # Split action into words to extract action type and parameters
    #     words = action.split()
    #     action_type = words[0]
    #     params = words[1:]

    #     # Copy the current state to avoid mutating the original
    #     new_state = set(state)

    #     if action_type == "pour":
    #         src_idx, tgt_idx, amt = [int(x) for x in params]
    #         src_contains = find_element(state, lambda x: x.split()[1] == str(src_idx))
    #         tgt_contains = find_element(state, lambda x: x.split()[1] == str(tgt_idx))

    #         src_r, src_g, src_b, src_amt = [int(x) for x in src_contains.split()[2:]]
    #         tgt_r, tgt_g, tgt_b, tgt_amt = [int(x) for x in tgt_contains.split()[2:]]

    #         # Calculate the actual amount to pour, which cannot exceed the source amount
    #         actual_amt = min(src_amt, amt)

    #         # Calculate the new color and amount for the target beaker
    #         if tgt_amt + actual_amt == 0:
    #             # If the target beaker is empty and no paint is poured, use the source color
    #             new_tgt_r, new_tgt_g, new_tgt_b = src_r, src_g, src_b
    #         else:
    #             # If the target beaker is not empty or paint is poured, calculate the new color using weighted average
    #             new_tgt_r = (tgt_r * tgt_amt + src_r * actual_amt) // (tgt_amt + actual_amt)
    #             new_tgt_g = (tgt_g * tgt_amt + src_g * actual_amt) // (tgt_amt + actual_amt)
    #             new_tgt_b = (tgt_b * tgt_amt + src_b * actual_amt) // (tgt_amt + actual_amt)

    #         new_tgt_amt = tgt_amt + actual_amt

    #         # Update the source beaker's amount
    #         new_src_amt = src_amt - actual_amt

    #         # Update the state with the new values
    #         new_state.remove(src_contains)
    #         new_state.remove(tgt_contains)
    #         new_state.add(f"contains {src_idx} {src_r} {src_g} {src_b} {new_src_amt}")
    #         new_state.add(f"contains {tgt_idx} {new_tgt_r} {new_tgt_g} {new_tgt_b} {new_tgt_amt}")

    #     return new_state
    
    # def state_transition(self, state, action):
    #     """
    #     This is the GROUND TRUTH state transition function!

    #     Applies an action to the current state and returns the new state.

    #     :param state: A set of predicates representing the current state.
    #     :param action: A string representing the action to be applied.
    #     :return: The new state as a set of predicates.
    #     """
        
    #     def find_element(my_set, condition):
    #         for element in my_set:
    #             if condition(element):
    #                 return element
    #         return None  

    #     # Split action into words to extract action type and parameters
    #     words = action.split()
    #     action_type = words[0]
    #     params = words[1:]

    #     # Copy the current state to avoid mutating the original
    #     new_state = set(state)
        
    #     if action_type == "pour":
    #         src_idx, tgt_idx, amt = [int(x) for x in params]
    #         src_contains = find_element(state, lambda x: x.split()[1] == str(src_idx))
    #         tgt_contains = find_element(state, lambda x: x.split()[1] == str(tgt_idx))

    #         src_r, src_g, src_b, src_amt = [int(x) for x in src_contains.split()[2:]]
    #         tgt_r, tgt_g, tgt_b, tgt_amt = [int(x) for x in tgt_contains.split()[2:]]

    #         transfer_amt = min(amt, src_amt)
    #         new_src_amt =  src_amt - transfer_amt
    #         new_tgt_amt = tgt_amt + transfer_amt
    #         new_tgt_color = SubtractiveModel.mix_colors(src_r, src_g, src_b, transfer_amt, tgt_r, tgt_g, tgt_b, tgt_amt)
    #         new_tgt_r, new_tgt_g, new_tgt_b = new_tgt_color
            
    #         new_src_contains = f"contains {src_idx} {src_r} {src_g} {src_b} {new_src_amt}"
    #         new_tgt_contains = f"contains {tgt_idx} {new_tgt_r} {new_tgt_g} {new_tgt_b} {new_tgt_amt}"

    #         new_state.discard(src_contains)
    #         new_state.discard(tgt_contains)

    #         new_state.add(new_src_contains)
    #         new_state.add(new_tgt_contains)
            
    #     return new_state
        


# LLM which acts as the world model builder
class WorldModelBuilder:
    def __init__(self, world_model : WorldModel, max_refine_iters=3):
        self.transitions = [] # transition bank
        self.world_model = world_model

        # string representation state transition function       
        self.stf_str = inspect.getsource(self.world_model.state_transition) 

        llm = ChatOpenAI(
            temperature=0,
            # model_name="gpt-3.5-turbo-1106"
            model_name='gpt-4-1106-preview'
        )
        self.llm = LLMChain(llm=llm, prompt=wmb_template)
        
        self.max_refine_iters = max_refine_iters 
        
    
    def refine_world_model(self, init=False, pass_rate_threshold=0.95) -> int:
        # test the existing world model
        results = self.get_test_results()
        results_str = self.filter_and_format_results(results)
        pass_rate = self.calculate_avg_sim_on_tc(results)

        best_stf_str = dedent(self.stf_str)
        best_pass_rate = pass_rate
        
        num_llm_queries = 0
        
        logging.info(f"Initial results:")
        logging.info(f" Results_str: {results_str}")
        logging.info(f" Average_sim_score: {pass_rate}")
        # breakpoint()
        
        for _ in range(self.max_refine_iters):
            # we can stop if all test cases pass:
            if pass_rate >= pass_rate_threshold:
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
            pass_rate = self.calculate_avg_sim_on_tc(results)
            
            # logging.info(f" new_results: {results_str}")
            logging.info(f" new sim_score after refining: {pass_rate}")

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
                "similarity" : rescaled_state_similarity(state, pred_next_state, next_state),
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
        
        sort_res = sorted(results, key = lambda x : x['similarity'])
        # breakpoint()
        
        for tc in sort_res:
            id = tc["id"]
            state = tc["state"]
            action = tc["action"]
            next_state, pred_next_state = tc["expected_out"], tc["actual_out"]

            result = "Pass" if pred_next_state == next_state else f"Fail ({tc['similarity']})"
            
            # only keep test cases which fail
            # if result == "Pass":
            #     continue
            

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
    
    def calculate_avg_sim_on_tc(self, results : list[dict]):
        if len(results) == 0:
            return 0

        # pass_cnt = 0
        total_sim_score = 0
        for tc in results:
            total_sim_score += rescaled_state_similarity(tc["state"], tc["expected_out"], tc["actual_out"])
        
        return total_sim_score / len(results)

    # def calculate_pass_rate(self, results : list[dict]):
    #     if len(results) == 0:
    #         return 0

    #     pass_cnt = 0
    #     for tc in results:
    #         if tc["expected_out"] == tc["actual_out"]:
    #             pass_cnt += 1
        
    #     return pass_cnt / len(results)
    
    def calculate_pass_rate_on_episode(self, episode_id):
        all_results = self.get_test_results()
        episode_results = []
        for res in all_results:
            if res["episode_id"] == episode_id:
                episode_results.append(res)
        return self.calculate_avg_sim_on_tc(episode_results)




def main():
    wmb = WorldModelBuilder()
    wmb.refine_world_model()

if __name__ == "__main__":
    main()

