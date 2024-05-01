from langchain.prompts import PromptTemplate

predicate_descriptions = """
contains ?x ?r ?g ?b ?a

- Used to indicate beaker ?x contains ?a ml of paint of color RGB: (?r, ?g, ?b)
"""

action_descriptions = """
pour ?x ?y ?a

- Pour ?a ml of paint from beaker x to beaker y
"""

env_description = """
Imagine an environment consisting of a collection of beakers each containing 
paint of a certain color and amount. The objective in the environment is to pour
paint from one beaker to another (mixing the paint) to achieve some target 
paint.
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

Previous state transition function:
{{prev_stf}}

Results from tester:
{{tester_res}}

First, for each test case that failed, you should try to explain the 
the difference between the predicted and actual next states

Then do some reasoning on how you should modify the existing state transition function
so that the predicted next_state matches the actual next_state.

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