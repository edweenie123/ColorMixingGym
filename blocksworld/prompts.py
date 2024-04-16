from langchain.prompts import PromptTemplate

predicate_descriptions = """
on ?x ?y 

- Used to indicate that block x is on top of block y

ontable ?x 

- Used to indicate that block x is on the top

clear ?x 

- Used to indicate that there is nothing on top of block x

handempty

- Used to indicate that your hand is empty

holding ?x 

- Used to indicate that you are holding block x
"""

# predicate_descriptions = """
# """

action_descriptions = """
pick-up ?x

- Pick up block x in your hand

put-down ?x

- Put the block x (that you are holding) on the table

stack ?x ?y

- Put block x (that you are holding) on top of block y

unstack ?x ?y

- Pick up block y which is currently on top of block x 
"""

env_description = """
Imagine an environment consisting of a collection of stackable blocks placed 
on a table. Each block can be stacked on top of another block or directly on the table, 
subject to certain rules and conditions. There is an agent in the environment
which can pick up blocks and stack blocks on top of each other.
"""

tips="""
- Do NOT add anything weird in the code section like ```python or any text which is not python code. 
- Also use set.discard instead of set.remove
- At the beginning of each if statement, you should extract the relevant elements from the params list.
- Please write the full state transition function and do NOT be lazy (do NOT say "implement the rest" or "add more if statements" in comments).
"""

wmb_str = f"""
Description of environment:
{env_description}

Here are a list of predicates:
{predicate_descriptions}

Here are a list of actions:
{action_descriptions} 

You are to write state transition function for this envirionmnt.

Previous state transition function.
{{prev_stf}}

Results from tester.
{{tester_res}}

First do some reasoning on how you should modify the existing state transition function
so that the predicted next_state matches the actual next_state.

In particular, for each test case that failed, you should try to explain why it failed.
You should use this format:

Test case i:

- Predicates that are in predicted next_state, but NOT in actual next_state
    - These are predicates to remove in the state transition function for the right action 
- Predicates that are in actual next_state but not in predicted next_state
    - These are predicates to add in the state transition function for the right action
- Which predicates to add / remove in the state transition function for which action
    
Then, output write FULL code (write the entire function) for the new state transition function.

Please sandwich the code you write for the state transition function with the markers

--- CODE START ---
<code here>
--- CODE END ---

Tips:
{tips}
"""




wmb_template = PromptTemplate(
    input_variables=["prev_stf", "tester_res"],
    template=wmb_str
)