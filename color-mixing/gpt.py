from openai import OpenAI
from env import *
import json
client = OpenAI()

def get_env_text_repr(obs, prev_plan):
    beaker_descriptions = []
    num_beakers = len(obs) - 1  # Excluding the target beaker

    # Process each beaker
    for idx in range(num_beakers):
        beaker = obs[idx]
        color_str = f"RGB({beaker[0]}, {beaker[1]}, {beaker[2]})"
        beaker_descriptions.append(f"Beaker {idx}: {color_str}, amount: {beaker[3]}ml")

    # Process the target beaker
    target_beaker = obs[-1]
    target_color_str = f"RGB({target_beaker[0]}, {target_beaker[1]}, {target_beaker[2]})"
    target_description = f"Target beaker: {target_color_str}, amount: {target_beaker[3]}ml"

    # Construct the query
    query = "Textual description of environment: \n" + "\n".join(beaker_descriptions) + "\n" + target_description

    if prev_plan == None:
        # first step 

        query += """\n
        Generate a numbered step by step plan in natural language to match the 
        target beaker as closely as possible (in both color and amount). 
        Some requirements in the plan: 
            - Each step must specify a "from beaker", a "to beaker" and an amount in ML
            - Try to predict the new contents (color and amount) of the to beaker after mixing (existing contents mixed with poured paint)
            - Make the plan at most 3 steps

        You should tell me which beaker you want to produce the target paint in. 
            - Explicitly calculate the amount of paint that will be in this beaker (after executing the plan) to make sure it matches the target beaker's amount
            - Also try to predict te color that will be produced in this beaker (after executing the plan) to make sure it matchs the target beaker's color
        """
    else:
        query += f"""\n
        Based on your previous plan:\n{prev_plan}
        \n
        After looking at this plan:
            - decide whether you want to revise the plan or keep the existing plan
            - if you choose to keep the existing plan, rewrite the exact same plan but WITHOUT THE FIRST STEP (because we already executed it)
        """

    query += "After this planning, output the new first move in the plan in JSON. If this is the last action in the plan, the is_done field should be True."

    return query

def extract_json_from_response(response_str):
    """
    Extracts a JSON object from the end of a string.
    """
    # Find the opening brace of the JSON object, assuming it's the last one in the string
     # Find the opening and closing braces of the last JSON object in the string
    end_index = response_str.rfind('}')
    if end_index == -1:
        return None

    start_index = response_str.rfind('{', 0, end_index)
    if start_index == -1:
        return None

    try:
        # Extract and parse the JSON object
        json_object = json.loads(response_str[start_index:end_index+1])
        return json_object
    except json.JSONDecodeError:
        # Return None if parsing fails
        return None


def extract_plan_from_response(response_str):
    end = response_str.rfind('{', 0)
    plan_str = response_str[:end]
    return plan_str


def test():
    system_prompt = """
    You are an agent in a "color-mixing environment". The environment consists 
    of several beakers, each containing a paint of a certain color and amount. 
    Your objective is to mix colors in beakers such that one beaker matches 
    a target beaker as close as possible (in both color and amount).

    At each timestep, you will be given a textual representation of environment
    which tells you the RGB value and amount of paints in each beaker as well
    as the target beaker you are trying to achieve.
    
    Note that the environment uses a subtractive color mixing model. You should 
    leverage this fact to predict the result of mixing two colors for the 
    purposes of planning. As you plan your steps, remember that when two colors 
    are mixed, the resulting color is influenced by the relative proportions of 
    each color. For instance, consider the mixing of red and blue liquids. If 
    you mix 50 ml of red with 10 ml of blue, the red color will have a 5 times 
    greater influence on the resulting color due to its higher volume. This 
    means the final color will lean more towards red. Keep this in mind while 
    planning your actions, as the proportions of colors you mix will 
    significantly affect the outcome.


    Given this description of the environment, your task is to generate an 
    action in a specific JSON format. Each action is a JSON object that 
    represents a move in the environment. The action should consist of the 
    following elements:

    - "from_beaker": An integer representing the index of the beaker from which the paint is poured.
    - "to_beaker": An integer representing the index of the beaker to which the paint is poured.
    - "transfer_amount": An integer (0 to 100) indicating how much paint (in ml) is transferred from the 'from_beaker' to the 'to_beaker'.
    - "is_done": A boolean value (true or false). Set to true if this is the final action and you want to compare the current state with the target. Set to false otherwise.
    - "compare_beaker": An integer representing the index of the beaker to compare with the target when 'is_done' is true. This field is only relevant if 'is_done' is true.

    First you should generate a multi-step plan of actions in order to achieve a 
    beaker (the compare beaker) which is as close as possible to the target 
    beaker (in both color and amount) in as few steps as possible. 

    After generating this plan, please output the first step in JSON according 
    to the given format. 
    """
    
    env = ColorMixingEnv(5, noise_level=0.1)

    # num_episodes = 5# You can adjust this number
    # for _ in range(num_episodes):
    obs, _ = env.reset()
    # print('Initial state:', obs)
    messages = [{"role": "system", "content": system_prompt}]
    # print(get_env_text_repr(obs))
    env.render()
    done = False

    prev_plan = None
    while not done:
        # convert observation to textual representation
        obs_text = get_env_text_repr(obs, prev_plan)
        print(obs_text)
        messages.append({"role": "user", "content": obs_text})

        # Query LLM for action
        response = client.chat.completions.create(
            # model="gpt-3.5-turbo-1106",  
            model="gpt-4",  
            # response_format={"type" : "json_object"},
            messages=messages
        )

        # Decode LLM response to action
        response = response.choices[0].message.content
        print(f'LLM response\n{response}')
        action_json = extract_json_from_response(response)
        prev_plan = extract_plan_from_response(response)
        
        from_idx = int(action_json['from_beaker'])
        to_idx = int(action_json['to_beaker'])
        transfer_amt = int(action_json['transfer_amount'])
        is_done = int(action_json['is_done'])
        if is_done:
            cmp_beaker = int(action_json['compare_beaker'])
        else:
            cmp_beaker = 0
        
        action = (from_idx, to_idx, transfer_amt, is_done, cmp_beaker)

        # action = decode_action(action_json, env.action_space)

        # Step in the environment
        obs, reward, done, truncated, info = env.step(action)
        print(f"Step: {env.step_count}, Action: {action}, Reward: {reward}")
        env.render()

# Close the rendering window
    if hasattr(env, 'close'):
        env.close()

if __name__ == '__main__':
    # env = ColorMixingEnv(5, noise_level=0.1)
    # obs, _ = env.reset()
    # print(get_env_text_repr(obs))
    test()
