from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from env import ColorMixingEnv
from abc import ABC, abstractmethod

import os
import pandas as pd
from prompts import *

def extract_section(response, section_name):
    """
    Extracts content sandwitched by
    --- <section_name> START ---
    --- <section_name> END ---
    """

    assert f'--- {section_name} START ---' in response and f'--- {section_name} END ---' in response, "Section markers not found in the string"
    # extract content
    content = response.split(f'--- {section_name} START ---')[1].split(f'--- {section_name} END ---')[0]
    
    return content

def extract_instructions(response):
    """
    Extracts instructions from plan in LLM response
    """
    # Extract the plan content
    plan_content = extract_section(response, 'PLAN')

    # Split the plan content into lines
    lines = plan_content.strip().split('\n')

    instructions = []

    for line in lines:
        # Ignore comments and empty lines
        if '#' in line:
            line = line.split('#')[0].strip()
        if line:
            # Split the line into action and parameters
            action, params = line.split('(')
            params = params.rstrip(')').split(',')
            params = [int(param.strip()) for param in params]  # Convert params to integers

            # Append action and parameters to the instructions list
            instructions.append((action.strip(), params))

    # Assert that the last action is DONE
    assert instructions[-1][0] == 'DONE', "Last action in the plan is not DONE"

    return instructions


class LLMAgent(ABC):
    @abstractmethod
    def next_action(self, obs: str):
        """Returns an action given the new observation"""
        pass

    def end_episode_reflection(self):
        pass

class UnreactivePlannerCritic(LLMAgent):
    def __init__(self):
        llm = ChatOpenAI(
            temperature=0,
            model_name='gpt-3.5-turbo-1106'
            # model_name='gpt-4-1106-preview'
        )

        self.planner = LLMChain(llm=llm, prompt=planner_template)
        self.critic = LLMChain(llm=llm, prompt=critic_template)
        self.inst_idx = 0 # instruction idx
        self.instructions = None # instructions
        # self.prev_plan = ''
        # self.plan_reasoning = ''
        self.feedback = ''
        self.planner_response = None
        self.critic_response = None
        self.trajectory = None # collect the trajectory in memory
        self.initial_state = None
        self.plan_reasoning = '' # stores the current plan_reasoning
        self.plan = '' # stores the current plan
        self.trajectory = '' # collect the trajectory here

    def next_action(self, obs: str):
        def find_next_execute():
            inst = self.instructions[self.inst_idx]
            self.inst_idx += 1

            # collect the trajectory
            if self.trajectory != '':
                # add the newest observation
                self.trajectory += obs + '\n'
            
            ins_type, args = inst
            ins_str = '{}({})'.format(ins_type, ', '.join(map(str, args)))
            if ins_type == 'POUR':
                self.trajectory += f'\nState after action {ins_str}:\n'
            elif ins_type == 'DONE':
                comp_idx, = args
                self.trajectory += f'\nCompare beaker {comp_idx} with target beaker'

            return inst
            
        if self.instructions != None:
            return find_next_execute()

        self.initial_state = obs 
        # don't have plan yet, so need to generate it
        print('Querying the planner...')
        planner_response = self.planner.run(
            state=obs,
            prev_reasoning=self.plan_reasoning,
            prev_plan=self.plan,
            feedback=self.feedback
        )
        print(f'planner_response={planner_response}')
        self.planner_response = planner_response
        self.plan_reasoning = extract_section(planner_response, 'REASONING')
        self.plan = extract_section(planner_response, 'PLAN') 

        self.instructions = extract_instructions(planner_response)
        self.inst_idx = 0
        
        return find_next_execute()
    
    def end_episode_reflection(self):

        print('Querying the critic...')
        critic_response = self.critic.run(
            initial_state=self.initial_state,
            plan_reasoning=self.plan_reasoning,
            plan=self.plan,
            trajectory=self.trajectory 
        )
        
        self.critic_response = critic_response
        print(f'critic_response={critic_response}')
        self.feedback = extract_section(critic_response, 'FEEDBACK')
        
        # need to reset a bunch of stuff
        self.trajectory = ''
        self.instructions = None


def iterative_control_loop(path, agent : LLMAgent, num_iter=3, render=True, log_path=None):
    env = ColorMixingEnv()
    env.load_state_from_file(path)
    env_state_text = env.get_env_text_repr()

    print(env_state_text)
    if render:
        env.render()

    # collect data
    df = pd.DataFrame(columns=[
        'iteration', 
        'color_score', 
        'amount_score', 
        'planner_response',
        'critic_response'
    ])

    for i in range(num_iter):
        while True:
            action = agent.next_action(env.get_env_text_repr())
            env.execute_instruction(action, render=render)
            
            if action[0] == 'DONE':
                break

        color_score, amount_score = env.calculate_scores()
        print(f'color_score = {color_score}, amount_score {amount_score}')

        if i == num_iter - 1:
            # no point in asking for feedback...
            if log_path != None:
                df.loc[len(df)] = (i, color_score, amount_score, agent.planner_response, None)
                df.to_csv(log_path, index=False)
            break 
        
        agent.end_episode_reflection()

        # reset the environment 
        env.load_state_from_file(path)
        
        # append to df 
        if log_path != None:
            df.loc[len(df)] = (i, color_score, amount_score, agent.planner_response, agent.critic_response)
            df.to_csv(log_path, index=False)

def replay(csv_path):
    df = pd.read_csv(csv_path)
    file_name = os.path.splitext(os.path.basename(csv_path))[0]

    env = ColorMixingEnv()
    env.load_state_from_file(f'data/level1/{file_name}')
    
    for _, row in df.iterrows():
        env.render()
        planner_response = row['planner_response']
        print(f'Planner response:\n{planner_response}')
        instructions = extract_instructions(planner_response)
        
        for inst in instructions:
            env.execute_instruction(inst, render=True)

        # execute_instructions(env, instructions)
        color_score, amount_score = row['color_score'], row['amount_score']
        print(f'color_score={color_score}, amount_score={amount_score}')

        critic_response = row['critic_response']
        print(f'Critic response:\n{critic_response}')

        # reset
        env.load_state_from_file(f'data/level1/{file_name}')

if __name__ == '__main__':
    replay('results/ablation/no_critic/beige.csv')
    # evaluate_on_dir('data/level1/')
    # agent = UnreactivePlannerCritic()
    # iterative_control_loop('data/level1/lavendar', agent)


    # for file in os.listdir('data/level1/'):
    #     print(f'Starting test for {file}')
    #     file_path = os.path.join('data/level1/', file)
        
    #     agent = UnreactivePlannerCritic()
    #     iterative_control_loop(file_path, agent, num_iter=3, render=True, log_path=f'results/tests/{file}.csv')


    # return
    # env = ColorMixingEnv()
    # llm = ChatOpenAI(
    #     temperature=0,
    #     # model_name='gpt-3.5-turbo-1106'
    #     model_name='gpt-4-1106-preview'
    # )
    # # planner = LLMChain(llm=llm, prompt=planner_template)
    # # critic = LLMChain(llm=llm, prompt=critic_template)

    # env.load_state_from_file('data/level1/beige.txt')
    # env.render()

    # while True:
    #     inp = input("enter next action: ") 
    #     a, b, c = inp.split(' ')
    #     env.step((int(a), int(b), int(c), 0, 0));
    #     print(get_env_text_repr(env._get_observation()))
    #     env.render()
    # exit()
        


    # print(get_env_text_repr(env._get_observation()))
    # env.render()

    # env.step((1, 5, 32, 0, 0));

    # print(get_env_text_repr(env._get_observation()))
    # env.render()

    # env.step((0, 5, 29, 0, 0));

    # print(get_env_text_repr(env._get_observation()))
    # env.render()



#     env_state_text = get_env_text_repr(env._get_observation())
#     print(env_state_text)
#     env.render()
#     response="""
#     --- PLAN START ---
# POUR(0, 5, 45) # pour 45 ml of red paint into beaker 5 to start the base magenta color
# POUR(2, 5, 45) # pour 45 ml of blue paint into beaker 5 to mix with the red
# # We now have 90ml of a lighter base magenta color in beaker 5.
# POUR(1, 5, 1)  # pour 1 ml of green paint to adjust the red and blue components
# # We have now 91ml of paint, leaving room for the addition of white to lighten the color.
# POUR(3, 5, 15) # pour 15 ml of white paint into beaker 5 to brighten the color
# # We now have 106ml of paint, which matches the target volume.
# # If the color is too light, we will add a very small amount of black to darken it slightly.
# # Since we don't have feedback, we'll assume the color is correct for now.
# DONE(5) # we assume the desired mixture is in beaker 5
# --- PLAN END ---
# """
#     instructions = extract_instructions(response)

#     # planner_response = planner.run(
#     #     state=env_state_text,
#     #     prev_reasoning='',
#     #     prev_plan='',
#     #     feedback=''
#     # )

#     trajectory = execute_instructions(env, instructions)

    exit() 
    
    initial_state = """
Beaker 0: Color: RGB(255, 0, 0), amount: 100ml
Beaker 1: Color: RGB(0, 255, 0), amount: 100ml
Beaker 2: Color: RGB(0, 0, 255), amount: 100ml
Beaker 3: Color: RGB(255, 255, 255), amount: 100ml
Beaker 4: Color: RGB(0, 0, 0), amount: 100ml
Beaker 5: Color: RGB(0, 0, 0), amount: 0ml
Target beaker: RGB(224, 28, 227), amount: 106ml
"""

    plan_reasoning = """
The target color is a shade of magenta, which is a mix of red and blue with a very small amount of green. The RGB values for the target are very high for red and blue, and very low for green. Since we're using a subtractive color mixing model, we need to mix the colors in such a way that we get the desired proportions.

Given that the target amount is 106ml, we need to figure out the proportions of each color to use. The red and blue values are almost equal, so we'll aim for a nearly 1:1 ratio of red to blue, with a very small amount of green.

To get the correct proportions, we can use the RGB values as a guide. The sum of the RGB values of the target is 224 + 28 + 227 = 479. To find the proportion of each color, we divide each RGB value by the total sum and then multiply by the target amount (106ml):

- Red: (224 / 479) * 106 ≈ 49.7ml
- Green: (28 / 479) * 106 ≈ 6.2ml
- Blue: (227 / 479) * 106 ≈ 50.1ml

Since we can only use whole numbers, we'll round these to the nearest whole number:

- Red: 50ml
- Green: 6ml
- Blue: 50ml

This adds up to 106ml, which is our target amount.

We can use beaker 5 as our mixing beaker. We'll start by pouring red and blue into beaker 5 to get the magenta color, and then we'll add a small amount of green to slightly adjust the color.

Let's start mixing!
"""
    plan = """
POUR(0, 5, 50) # pour 50 ml of red paint into beaker 5
POUR(2, 5, 50) # pour 50 ml of blue paint into beaker 5
POUR(1, 5, 6)  # pour 6 ml of green paint into beaker 5
DONE(5)        # the desired mixture is in beaker 5
"""

    trajectory = """
State after action POUR(0, 5, 50):
Beaker 0: Color: RGB(255, 0, 0), amount: 50ml
Beaker 1: Color: RGB(0, 255, 0), amount: 100ml
Beaker 2: Color: RGB(0, 0, 255), amount: 100ml
Beaker 3: Color: RGB(255, 255, 255), amount: 100ml
Beaker 4: Color: RGB(0, 0, 0), amount: 100ml
Beaker 5: Color: RGB(255, 0, 0), amount: 50ml
Target beaker: RGB(224, 28, 227), amount: 106ml

State after action POUR(2, 5, 50):
Beaker 0: Color: RGB(255, 0, 0), amount: 50ml
Beaker 1: Color: RGB(0, 255, 0), amount: 100ml
Beaker 2: Color: RGB(0, 0, 255), amount: 50ml
Beaker 3: Color: RGB(255, 255, 255), amount: 100ml
Beaker 4: Color: RGB(0, 0, 0), amount: 100ml
Beaker 5: Color: RGB(128, 0, 128), amount: 100ml
Target beaker: RGB(224, 28, 227), amount: 106ml

State after action POUR(1, 5, 6):
Beaker 0: Color: RGB(255, 0, 0), amount: 50ml
Beaker 1: Color: RGB(0, 255, 0), amount: 94ml
Beaker 2: Color: RGB(0, 0, 255), amount: 50ml
Beaker 3: Color: RGB(255, 255, 255), amount: 100ml
Beaker 4: Color: RGB(0, 0, 0), amount: 100ml
Beaker 5: Color: RGB(121, 15, 121), amount: 106ml
Target beaker: RGB(224, 28, 227), amount: 106ml

Compare beaker 5 with target beaker
"""
    llm = ChatOpenAI(
        temperature=0,
        # model_name='gpt-3.5-turbo-1106'
        model_name='gpt-4-1106-preview'
    )

    # planner = LLMChain(llm=llm, prompt=planner_template)
    critic = LLMChain(llm=llm, prompt=critic_template)
    
    feedback = critic.run(
        initial_state=initial_state,
        plan_reasoning=plan_reasoning,
        plan=plan,
        trajectory=trajectory 
    )
    
    print(feedback)


    # env = ColorMixingEnv()

    # env.load_state_from_file('data/level1/beige.txt')
    # env.render()

    # obs, reward, done, trunacated, _ = env.step((0, 5, 20, 0, 3))
    # env.render()
    # obs, reward, done, trunacated, _ = env.step((1, 5, 20, 0, 3))
    # # print('step 1:', obs, reward, env.calculate_score(3))
    # env.render()
    # obs, reward, done, trunacated, _ = env.step((2, 5, 20, 0, 3))
    # env.render()

    # obs, reward, done, trunacated, _ = env.step((0, 5, 15, 0, 3))
    # env.render()
    # obs, reward, done, trunacated, _ = env.step((3, 5, 100, 0, 3))
    # env.render()
    # obs, reward, done, trunacated, _ = env.step((0, 5, 20, 0, 3))
    # # print('step 1:', obs, reward, env.calculate_score(3))
    # env.render()
    # obs, reward, done, trunacated, _ = env.step((1, 5, 15, 0, 3))
    # env.render()

    # obs, reward, done, trunacated, _ = env.step((0, 5, 15, 0, 3))
    # env.render()


    # llm = ChatOpenAI(
    #     temperature=0,
    #     # model_name='gpt-3.5-turbo-1106'
    #     model_name='gpt-4-1106-preview'
    # )
    
    # planner = LLMChain(llm=llm, prompt=planner_template)
    # env_state_text = get_env_text_repr(env._get_observation())
    # response = planner.run(env_state_text)
    
    # # print(response)
    # df = pd.read_csv('results/l1-no-refine.csv')
    # response = list(df.loc[df['name'] == 'beige.txt']['llm_response'])[0]
    # print(response)
    # instructions = extract_instructions(response)
    # print(f'instructions={instructions}')
    # # execute_instructions(env, instructions)
    
    # print(env.calculate_scores())
    


    
    # print(extract_instructions(more_examples))


    # env.render()
    # # print('initial score:', env.calculate_score(3))
    # # # input()
    # obs, reward, done, trunacated, _ = env.step((1, 3, 50, 0, 3))
    # env.render()
    # obs, reward, done, trunacated, _ = env.step((0, 3, 44, 0, 3))
    # # print('step 1:', obs, reward, env.calculate_score(3))
    # env.render()
    # obs, reward, done, trunacated, _ = env.step((2, 3, 26, 0, 3))
    # env.render()

    # obs, _ = env.reset()
    
    # print(get_env_text_repr(obs))
    # env.render()

    # env.render()
    # print('initial score:', env.calculate_score(3))

    # # input()
    # env.render()
    # obs, reward, done, trunacated, _ = env.step((0, 3, 75, 0, 3))
    # print('step 1:', obs, reward, env.calculate_score(3))
    # env.render()

