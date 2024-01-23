from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from env import ColorMixingEnv
import os
import pandas as pd
from prompts import *

def get_env_text_repr(obs):
    """
    Receives observation from environment (env state) and converts it
    to a string representation
    """
    beaker_descriptions = []
    num_beakers = len(obs) - 1  # Excluding the target beaker

    # Process each beaker
    for idx in range(num_beakers):
        beaker = obs[idx]
        color_str = f"Color: RGB({beaker[0]}, {beaker[1]}, {beaker[2]})"
        beaker_descriptions.append(f"Beaker {idx}: {color_str}, amount: {beaker[3]}ml")

    # Process the target beaker
    target_beaker = obs[-1]
    target_color_str = f"RGB({target_beaker[0]}, {target_beaker[1]}, {target_beaker[2]})"
    target_description = f"Target beaker: {target_color_str}, amount: {target_beaker[3]}ml"

    text_repr = "\n".join(beaker_descriptions) + "\n" + target_description
    return text_repr

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

def execute_instructions(env : ColorMixingEnv, instructions, render=True):
    """
    returns trajectory
    """
    if render:
        env.render()
    
    trajectory = ""

    for instruction in instructions:
        ins_type, args = instruction
        ins_str = '{}({})'.format(ins_type, ', '.join(map(str, args)))
        
        if ins_type == 'POUR':
            from_idx, to_idx, amt = args
            env.step((from_idx, to_idx, amt, 0, 0))

            trajectory += f'\nState after action {ins_str}:\n'
            trajectory += get_env_text_repr(env._get_observation()) + '\n'

        elif ins_type == 'DONE':
            comp_idx, = args
            env.step((0, 0, 0, 1, comp_idx))
            
            trajectory += f'\nCompare beaker {comp_idx} with target beaker'

        if render:
            env.render()
    
    return trajectory

def evaluate_on_dir(dir_path, render=False):
    env = ColorMixingEnv(
        num_beakers=4
    )

    llm = ChatOpenAI(
        temperature=0,
        # model_name='gpt-3.5-turbo-1106'
        model_name='gpt-4-1106-preview'
    )

    planner = LLMChain(llm=llm, prompt=planner_template)
    critic = LLMChain(llm=llm, prompt=critic_template)
    
    df = pd.DataFrame(columns=['name', 'color_score', 'amount_score', 'llm_response'])
    for file in os.listdir(dir_path):
        print(f'Starting test for {file}')
        file_path = os.path.join(dir_path, file)
        env.load_state_from_file(file_path)
        
        if render:
            env.render()
        
        initial_state_repr = get_env_text_repr(env._get_observation())
        print(initial_state_repr)
        response = planner.run(initial_state_repr)
        print(response)

        instructions = extract_instructions(response)
        trajectory = execute_instructions(env, instructions, render=render)

        # feedback = planner()

        color_score, amount_score = env.calculate_scores()
        # append to df 
        df.loc[len(df)] = (file, color_score, amount_score, response)
    
        df.to_csv('results/l1-no-refine.csv', index=False)

# def generate_episode(path):
#     env = ColorMixingEnv()
#     llm = ChatOpenAI(
#         temperature=0,
#         # model_name='gpt-3.5-turbo-1106'
#         model_name='gpt-4-1106-preview'
#     )
#     planner = LLMChain(llm=llm, prompt=planner_template)
#     llm.cal
#     # critic = LLMChain(llm=llm, prompt=critic_template)

#     env.load_state_from_file(path)
#     env_state_text = get_env_text_repr(env._get_observation())
#     print(env_state_text)
#     env.render()
    



def test_on_single(path, num_iter, render=True):
    env = ColorMixingEnv()
    llm = ChatOpenAI(
        temperature=0,
        # model_name='gpt-3.5-turbo-1106'
        model_name='gpt-4-1106-preview'
    )
    planner = LLMChain(llm=llm, prompt=planner_template)
    critic = LLMChain(llm=llm, prompt=critic_template)

    env.load_state_from_file(path)
    env_state_text = get_env_text_repr(env._get_observation())
    print(env_state_text)
    if render:
        env.render()

    prev_plan = ''
    prev_reasoning = ''
    feedback = ''

    df = pd.DataFrame(columns=[
        'iteration', 
        'color_score', 
        'amount_score', 
        'planner_response',
        'critic_response'
    ])

    for i in range(num_iter):
        print('Querying the planner...')
        planner_response = planner.run(
            state=env_state_text,
            prev_reasoning=prev_reasoning,
            prev_plan=prev_plan,
            feedback=feedback
        )
        print(f'planner_response=\n{planner_response}')

        instructions = extract_instructions(planner_response)
        plan_reasoning = extract_section(planner_response, 'REASONING')
        plan = extract_section(planner_response, 'PLAN') 

        trajectory = execute_instructions(env, instructions, render=render)
        color_score, amount_score = env.calculate_scores()
        print(f'color_score = {color_score}, amount_score {amount_score}')

        if i == num_iter - 1:
            # no point in asking for feedback...
            df.loc[len(df)] = (i, color_score, amount_score, planner_response, None)
            df.to_csv(f'results/level1/{os.path.basename(path)}.csv', index=False)
            break 

        print('Querying the critic...')
        critic_response = critic.run(
            initial_state=env_state_text,
            plan_reasoning=plan_reasoning,
            plan=plan,
            trajectory=trajectory 
        )
        print(f'critic_response={critic_response}')
    
        feedback = extract_section(critic_response, 'FEEDBACK')
        prev_reasoning = plan_reasoning
        prev_plan = plan

        # reset the environment 
        env.load_state_from_file(path)
        
        # append to df 
        df.loc[len(df)] = (i, color_score, amount_score, planner_response, critic_response)
        df.to_csv(f'results/level1/{os.path.basename(path)}.csv', index=False)

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
        execute_instructions(env, instructions)
        color_score, amount_score = row['color_score'], row['amount_score']
        print(f'color_score={color_score}, amount_score={amount_score}')

        critic_response = row['critic_response']
        print(f'Critic response:\n{critic_response}')

        # reset
        env.load_state_from_file(f'data/level1/{file_name}')

if __name__ == '__main__':
    replay('results/level1/magenta.csv')
    # evaluate_on_dir('data/level1/')

    # test_on_single('data/level1/magenta', 3)
    # for file in os.listdir('data/level1/'):
    #     print(f'Starting test for {file}')
    #     file_path = os.path.join('data/level1/', file)
    #     test_on_single(file_path, 3, render=False)


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

