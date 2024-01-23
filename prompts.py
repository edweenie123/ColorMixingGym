from langchain.prompts import PromptTemplate

planner_instructions = """
Imagine you are in a "color-mixing environment". The environment consists of

- Initial starting beakers each containing a paint of a certain color and amount. 
- A target beaker which you want to create

Your objective is to plan a sequence of actions in order to produce a beaker 
that matches the target beaker as close as possible (in both color and amount).

You will be given a textual representation of environment which tells you the 
RGB value and amount of paints in each beaker as well as the target beaker you 
are trying to achieve.

After receiving the environment state, you should first do some reasoning in 
natural language to reason how exactly you are going to produce the target beaker.
Then, you should output a sequence of actions. There are 2 types of actions:

1. POUR(from_beaker, to_beaker, transfer_amount)
2. DONE(beaker)

A POUR action is used to pour paint from one beaker where

- "from_beaker" is an integer representing the index of the beaker [0, N-1] from which the paint is poured from.
- "to_beaker" is an integer representing the index of the beaker [0, N-1] which the paint is poured to.
- "transfer_amount" is an integer (0 to 100) indicating how much paint (in ml) is transferred from the 'from_beaker' to the 'to_beaker'.

A DONE action indicates the end of a plan where

- "beaker" is an integer representing the index of the beaker which contains desired mixture

You may also be given  
1. a previous plan with reasoning
2. feedback for that plan

In this case, you should start with the previous plan and incorporate the 
feedback to improve the plan instead of starting from scratch.

Here is are some examples:

--- ENVIRONMENT START ---
ENVIRONMENT STATE:
Beaker 0: Color: RGB(255, 0, 0), amount: 97ml
Beaker 1: Color: RGB(0, 255, 0), amount: 98ml
Beaker 2: Color: RGB(0, 0, 255), amount: 101ml
Beaker 3: Color: RGB(0, 0, 0), amount: 0ml
Target beaker: RGB(76, 107, 43), amount: 120ml
--- ENVIRONMENT END ---

--- PREVIOUS PLAN START ---
Plan reasoning: 

Plan:

Feedback:
--- PREVIOUS PLAN END ---

--- REASONING START ---
Since there is no previous plan, I will formula a plan from scratch.
I see that I have access to a good amount of red, green and blue paint,
relevant to the target beaker so paint shortage shouldn't be a problem.
I also notice that beaker 3 is empty so I will use this beaker to store
my final result. The target color seems to be a woodland (darkish) green color.
Judging from the RGB code of the target beaker, I think I will need the most
green paint (from beaker 1), a good amount of red paint (from beaker 0)
and a bit of blue paint (from beaker 3). As an initial plan, lets say
I add 50ml of green paint, 44ml of red paint and 26ml of blue paint
to the empty beaker as 50 + 44 + 26 = 120ml which is the target amount.
--- REASONING END ---

--- PLAN START ---
POUR(1, 3, 50) # pour 50 ml of red paint to empty beaker 3
POUR(0, 3, 44) # pour 44 ml of green paint into beaker 3
POUR(2, 3, 26) # pour 26 ml of blue paint into beaker 3
DONE(3) # desired mixture was created in beaker 3
--- PLAN END ---

END OF EXAMPLES

Before you start here are some tips:

- You MUST sandwitch each section with tokens --- <SECTION NAME> START --- and --- <SECTION NAME> END ---
- You can use the # symbol to write comments in the plan
- Do NOT put mathematical expressions in any of the parameters to POUR. Just put integers.
- The environment uses a subtractive color mixing model. You should 
leverage this fact to predict the result of mixing two colors for the 
purposes of planning. 
- As you plan your steps, remember that when two colors are mixed, the resulting 
color is influenced by the relative proportions of each color. For instance, 
consider the mixing of red and blue liquids. If you mix 50 ml of red with 10 ml 
of blue, the red color will have a 5 times greater influence on the resulting 
color due to its higher volume. This means the final color will lean more towards red. 
Keep this in mind while planning your actions, as the proportions of colors you mix will 
significantly affect the outcome.
- When generating a new plan, you SHOULD NOT continue off the previous plan; 
This means that you should output ALL steps required to produce the target beaker
from the INITIAL evironment state!

--- ENVIRONMENT START ---
{state}
--- ENVIRONMENT END ---

--- PREVIOUS PLAN START ---
Plan reasoning: 
{prev_reasoning}

Plan:
{prev_plan}

Feedback:
{feedback}
--- PREVIOUS PLAN END ---
"""




planner_reactive_instructions = """
Imagine you are in a "color-mixing environment". The environment consists of

- Initial starting beakers each containing a paint of a certain color and amount. 
- A target beaker which you want to create

Your objective is to plan a sequence of actions in order to produce a beaker 
that matches the target beaker as close as possible (in both color and amount).

You will be given a textual representation of environment which tells you the 
RGB value and amount of paints in each beaker as well as the target beaker you 
are trying to achieve.

After receiving the environment state, you should first do some reasoning in 
natural language to reason how exactly you are going to produce the target beaker.
Then, you should output a sequence of actions. There are 2 types of actions:

1. POUR(from_beaker, to_beaker, transfer_amount)
2. DONE(beaker)

A POUR action is used to pour paint from one beaker where

- "from_beaker" is an integer representing the index of the beaker [0, N-1] from which the paint is poured from.
- "to_beaker" is an integer representing the index of the beaker [0, N-1] which the paint is poured to.
- "transfer_amount" is an integer (0 to 100) indicating how much paint (in ml) is transferred from the 'from_beaker' to the 'to_beaker'.

A DONE action indicates the end of a plan where

- "beaker" is an integer representing the index of the beaker which contains desired mixture

After generating the plan, the first action in the plan will be executed
and you will be provided with the new environment state.

First do some reasoning to decide whether you want to keep the existing plan
or revise the plan according to the new environment state.

Then, output the revised (or the same plan) according to your reasoning.

Here is an example:

--- STATE 0 START ---
Initial state:
Beaker 0: Color: RGB(255, 0, 0), amount: 100ml
Beaker 1: Color: RGB(0, 255, 0), amount: 100ml
Beaker 2: Color: RGB(0, 0, 255), amount: 100ml
Beaker 3: Color: RGB(255, 255, 255), amount: 100ml
Beaker 4: Color: RGB(0, 0, 0), amount: 100ml
Beaker 5: Color: RGB(0, 0, 0), amount: 0ml
Target beaker: RGB(137, 153, 255), amount: 135ml
--- STATE 0 END ---

--- REASONING 0 START ---
The target color is a light blue with a hint of red and green. Since the target color has the highest value in blue (255), followed by green (153), and the least in red (137), I will need to mix the primary colors in such a way that blue is the most dominant, followed by green, and then red.

Given that the target amount is 135ml, I need to calculate the proportions of each color to match the target RGB values. To simplify the calculations, I will first normalize the RGB values of the target color to a total of 135 units. The normalized values are approximately:

- Red: (137 / (137 + 153 + 255)) * 135 = 28.5ml
- Green: (153 / (137 + 153 + 255)) * 135 = 31.9ml
- Blue: (255 / (137 + 153 + 255)) * 135 = 74.6ml

Since I can only use whole numbers, I will round these to the nearest whole number:

- Red: 29ml
- Green: 32ml
- Blue: 74ml

This adds up to 135ml, which is the target amount. I will use beaker 5 as my mixing beaker. I will start by pouring blue, then green, and finally red into beaker 5. If the color is not exactly as desired, I can adjust by adding small amounts of white or black paint to lighten or darken the color respectively.

Now, let's create the plan.
--- REASONING 0 END ---

--- PLAN 0 START ---
POUR(2, 5, 74) # pour 74 ml of blue paint into beaker 5
POUR(1, 5, 32) # pour 32 ml of green paint into beaker 5
POUR(0, 5, 29) # pour 29 ml of red paint into beaker 5
# Check the color, if it needs adjustment, use white or black paint
# Assuming the color is as desired, we are done
DONE(5) # the desired mixture is created in beaker 5
--- PLAN 0 END ---

--- STATE 1 START ---
State after POUR(2, 5, 74):
Beaker 0: Color: RGB(255, 0, 0), amount: 100ml
Beaker 1: Color: RGB(0, 255, 0), amount: 100ml
Beaker 2: Color: RGB(0, 0, 255), amount: 26ml
Beaker 3: Color: RGB(255, 255, 255), amount: 100ml
Beaker 4: Color: RGB(0, 0, 0), amount: 100ml
Beaker 5: Color: RGB(0, 0, 255), amount: 74ml
Target beaker: RGB(254, 222, 197), amount: 130ml
--- STATE 1 END ---

--- REASONING 1 START ---
It looks like 74ml of blue paint was successfull transfered to beaker 5.
Beaker 5 is now pure blue paint. It looks like nothing went wrong
so far so I will stick with the same plan.

The next steps in the plan are to pour 32 ml of green paint and then mour 29 
ml of red paint.
--- REASONING 1 END ---

--- PLAN 1 START ---
# DON't include FIRST step in plan because we already executed it
POUR(1, 5, 32) # pour 32 ml of green paint into beaker 5
POUR(0, 5, 29) # pour 29 ml of red paint into beaker 5
DONE(5) # the desired mixture is created in beaker 5
--- PLAN 1 END ---

--- STATE 2 START ---
State after POUR(1, 5, 32):
Beaker 0: Color: RGB(255, 0, 0), amount: 100ml
Beaker 1: Color: RGB(0, 255, 0), amount: 68ml
Beaker 2: Color: RGB(0, 0, 255), amount: 26ml
Beaker 3: Color: RGB(255, 255, 255), amount: 100ml
Beaker 4: Color: RGB(0, 0, 0), amount: 100ml
Beaker 5: Color: RGB(0, 77, 179), amount: 106ml
Target beaker: RGB(254, 222, 197), amount: 130ml
--- STATE 2 END ---

--- REASONING 2 START ---
It looks like 32ml of green paint was successfull transferred to beaker 5.
The resulting mixture looks a bit dark. But it is okay for now. 
We will continue with the same plan. The next of the plan is to pour 29 ml 
of red paint into beaker 5.
--- REASONING 2 END ---

--- PLAN 2 START ---
POUR(0, 5, 29) # pour 29 ml of red paint into beaker 5
DONE(5) # the desired mixture is created in beaker 5
--- PLAN 2 END ---

--- STATE 2 START ---
State after POUR(0, 5, 29):
Beaker 0: Color: RGB(255, 0, 0), amount: 71ml
Beaker 1: Color: RGB(0, 255, 0), amount: 68ml
Beaker 2: Color: RGB(0, 0, 255), amount: 26ml
Beaker 3: Color: RGB(255, 255, 255), amount: 100ml
Beaker 4: Color: RGB(0, 0, 0), amount: 100ml
Beaker 5: Color: RGB(55, 61, 141), amount: 135ml
Target beaker: RGB(137, 153, 255), amount: 135ml
--- STATE 2 END ---

--- REASONING 3 START ---
It looks like 29ml of red paint was successfully added to beaker 5.
The next step of the plan compares beaker 5 with the target beaker 5.
The amount of 135ml produced in beaker 5 is good (same as target).
The ratios of the red, green and blue components within beaker 5
is similar to the target beaker as well.

The biggest problem is that the color in beaker 5 is 
no where near bright enough. The good news is that we 
have access to a lot of white paint in beaker 3
which could use to brighten the mixture in 5.

However, this will cause us to go above the target amount of 135ml
so we will have to pour some liquid out.

Lets pour this liquid out into beaker 4 as we probabily won't need black
paint anytime soon.
--- REASONING 3 END ---

--- PLAN 3 START ---
POUR(3, 5, 50) # pour some white paint into beaker 5
POUR(5, 4, 50) # pour excess paint into black beaker
DONE(5) # the desired mixture is created in beaker 5
--- PLAN 3 END ---

END OF EXAMPLE

Before you start here are some tips:

- You MUST sandwitch each section with tokens --- <SECTION NAME> START --- and --- <SECTION NAME> END ---
- When you are given STATE x, you should ONLY output REASONING x and PLAN x (nothing else)
- You can use the # symbol to write comments in the plan
- Do NOT put mathematical expressions in any of the parameters to POUR. Just put integers.
- The environment uses a subtractive color mixing model. You should 
leverage this fact to predict the result of mixing two colors for the 
purposes of planning. 
- As you plan your steps, remember that when two colors are mixed, the resulting 
color is influenced by the relative proportions of each color. For instance, 
consider the mixing of red and blue liquids. If you mix 50 ml of red with 10 ml 
of blue, the red color will have a 5 times greater influence on the resulting 
color due to its higher volume. This means the final color will lean more towards red. 
Keep this in mind while planning your actions, as the proportions of colors you mix will 
significantly affect the outcome.
"""


critic_instructions = """
You are a critic whose objective is to provide valuable feedback to a planner
for a "color-mixing environment". 

The environment consists of several beakers each with paint of a certain color 
and amount. The objective of the planner is to mix paints to create some 
specified target beaker.

There are 2 types of actions:

1. POUR(from_beaker, to_beaker, transfer_amount)
2. DONE(beaker)

A POUR action is used to pour paint from one beaker where

- "from_beaker" is an integer representing the index of the beaker [0, N-1] from which the paint is poured from.
- "to_beaker" is an integer representing the index of the beaker [0, N-1] which the paint is poured to.
- "transfer_amount" is an integer (0 to 100) indicating how much paint (in ml) is transferred from the 'from_beaker' to the 'to_beaker'.

A DONE action indicates the end of a plan where

- "beaker" is an integer representing the index of the beaker which contains desired mixture

You will be given:
1. The initial environment state
2. The planner's reasoning
3. The planner's generated plan
4. The environment state after each action in the plan (the trajectory)

With all this information you should first do some reasoning to try to understand
what the planner's strategy and understand what happened during the trajectory.

After doing this reasoning, please output concrete feedback to the planner
to help it improve its plan.

Here is an example:
    
--- ENVIRONMENT START ---
Beaker 0: Color: RGB(255, 0, 0), amount: 100ml
Beaker 1: Color: RGB(0, 255, 0), amount: 100ml
Beaker 2: Color: RGB(0, 0, 255), amount: 100ml
Beaker 3: Color: RGB(255, 255, 255), amount: 100ml
Beaker 4: Color: RGB(0, 0, 0), amount: 100ml
Beaker 5: Color: RGB(0, 0, 0), amount: 0ml
Target beaker: RGB(137, 153, 255), amount: 135ml
--- ENVIRONMENT END ---

--- PLAN REASONING START ---
The target color is a light blue with a hint of red and green. Since the target color has the highest value in blue (255), followed by green (153), and the least in red (137), I will need to mix the primary colors in such a way that blue is the most dominant, followed by green, and then red.

Given that the target amount is 135ml, I need to calculate the proportions of each color to match the target RGB values. To simplify the calculations, I will first normalize the RGB values of the target color to a total of 135 units. The normalized values are approximately:

- Red: (137 / (137 + 153 + 255)) * 135 = 28.5ml
- Green: (153 / (137 + 153 + 255)) * 135 = 31.9ml
- Blue: (255 / (137 + 153 + 255)) * 135 = 74.6ml

Since I can only use whole numbers, I will round these to the nearest whole number:

- Red: 29ml
- Green: 32ml
- Blue: 74ml

This adds up to 135ml, which is the target amount. I will use beaker 5 as my mixing beaker. I will start by pouring blue, then green, and finally red into beaker 5. If the color is not exactly as desired, I can adjust by adding small amounts of white or black paint to lighten or darken the color respectively.

Now, let's create the plan.
--- PLAN REASONING END ---

--- PLAN START ---
POUR(2, 5, 74) # pour 74 ml of blue paint into beaker 5
POUR(1, 5, 32) # pour 32 ml of green paint into beaker 5
POUR(0, 5, 29) # pour 29 ml of red paint into beaker 5
# Check the color, if it needs adjustment, use white or black paint
# Assuming the color is as desired, we are done
DONE(5) # the desired mixture is created in beaker 5

--- PLAN END ---

--- TRAJECTORY START ---
State after action POUR(2, 5, 74):
Beaker 0: Color: RGB(255, 0, 0), amount: 100ml
Beaker 1: Color: RGB(0, 255, 0), amount: 100ml
Beaker 2: Color: RGB(0, 0, 255), amount: 26ml
Beaker 3: Color: RGB(255, 255, 255), amount: 100ml
Beaker 4: Color: RGB(0, 0, 0), amount: 100ml
Beaker 5: Color: RGB(0, 0, 255), amount: 74ml
Target beaker: RGB(137, 153, 255), amount: 135ml

State after action POUR(1, 5, 32):
Beaker 0: Color: RGB(255, 0, 0), amount: 100ml
Beaker 1: Color: RGB(0, 255, 0), amount: 68ml
Beaker 2: Color: RGB(0, 0, 255), amount: 26ml
Beaker 3: Color: RGB(255, 255, 255), amount: 100ml
Beaker 4: Color: RGB(0, 0, 0), amount: 100ml
Beaker 5: Color: RGB(0, 77, 179), amount: 106ml
Target beaker: RGB(137, 153, 255), amount: 135ml

State after action POUR(0, 5, 29):
Beaker 0: Color: RGB(255, 0, 0), amount: 71ml
Beaker 1: Color: RGB(0, 255, 0), amount: 68ml
Beaker 2: Color: RGB(0, 0, 255), amount: 26ml
Beaker 3: Color: RGB(255, 255, 255), amount: 100ml
Beaker 4: Color: RGB(0, 0, 0), amount: 100ml
Beaker 5: Color: RGB(55, 61, 141), amount: 135ml
Target beaker: RGB(137, 153, 255), amount: 135ml

Compare beaker 5 with target beaker
--- TRAJECTORY END ---

--- FEEDBACK REASONING START ---
The produced mixture in beaker 5 is looks like

Color: RGB(55, 61, 141), amount: 135ml 

whereas the target beaker looks like 

Target beaker: RGB(137, 153, 255), amount: 135ml.

The produced mixture's amount is perfect, but the color is quite far away.
Judging from the planner's reasoning, they used ratios to determine
the appropriate amounts of red, blue and green paint to add to the 
mixture. This is good as the ratios of the RGB components in the 
produced mixture (55, 61, 141) is very close to the target mixture (137, 153, 255).

The main issue is that the produced mixture is not bright enough.
Since we have access to 100ml of white paint in beaker 4 that was not used
at all, we can make use of this white paint to make the mixture brighter!

Let us use a huge amount of white paint (perhaps 50 ml) to see what happens.

If this turns out to be too much white paint, we can adjust it in the
next planning iteration.

--- FEEDBACK REASONING END ---

--- FEEDBACK START ---
Good things you did:
1. The relative ratios of red, green and blue paint in the produced mixture are good
    - You did a good job calculating ratios within your reasoning prior to the plan
2. The final amount produced 135 ml is good

Things to improve:
1. The produced color RGB(55, 61, 141) is no where near bright enough compared to the target beaker
    - I would suggest you add a substantial amount of white (maybe 100 ml of white to the mixture) initially
    and then add the correct ratios of red, green and blue (if this is too much white we can reduce it in the next iteration)
--- FEEDBACK END ---

END OF EXAMPLES.

Before you start here are some tips:
- You should do some feedback reasoning before actually providing the final feedback
    - To to ensure that the final feedback you give is concise and high quality
- You MUST sandwitch your feedback reasoning with tokens --- FEEDBACK REASONING START --- and --- FEEDBACK REASONING END ---
- You MUST sandwitch your feedback with tokens --- FEEDBACK START --- and --- FEEDBACK END ---
- Your feedback should be as specific and helpful as possible to improve the plan in the next iteration
    - Mention good things about the plan (aspects to keep in the next iteration)
    - Also mention bad things about the plan (and how to fix them!)
    - Mention specific ml amounts in your feedback!

--- ENVIRONMENT START ---
{initial_state}
--- ENVIRONMENT END ---

--- REASONING START ---
{plan_reasoning}
--- REASONING END ---

--- PLAN START ---
{plan}
--- PLAN END ---

--- TRAJECTORY START ---
{trajectory}
--- TRAJECTORY END ---
"""

more_examples = """
Beaker 0: Color: RGB(255, 0, 0), amount: 100ml
Beaker 1: Color: RGB(0, 255, 0), amount: 100ml
Beaker 2: Color: RGB(0, 0, 255), amount: 100ml
Beaker 3: Color: RGB(0, 0, 0), amount: 0ml
Target beaker: RGB(62, 25, 48), amount: 86ml
--- REASONING START ---
The target color is a shade of purple with a higher amount of red, followed by blue, and a smaller amount of green. Since the target amount is 86ml, I need to calculate the proportions of each primary color to achieve the target RGB values.

The RGB values of the target beaker are (62, 25, 48). To achieve this, I need to mix the colors in a way that the red component is the highest, followed by blue, and then green. I will use the empty beaker (beaker 3) to mix the colors.

To get the proportions right, I can use the RGB values as a guide. The sum of the RGB values is 62 + 25 + 48 = 135. To find the percentage of each color, I divide each RGB value by the total sum and then multiply by the target amount (86ml):

- Red: (62 / 135) * 86 ≈ 39ml
- Green: (25 / 135) * 86 ≈ 16ml
- Blue: (48 / 135) * 86 ≈ 31ml

These are approximate values, and I will round them to the nearest whole number for the pouring actions. The sum of these amounts is 39 + 16 + 31 = 86ml, which matches the target amount.

Now, I will create a plan to mix these amounts from the starting beakers into the empty beaker.
--- REASONING END ---

--- PLAN START ---
POUR(0, 3, 39) # pour 39 ml of red paint into beaker 3
POUR(1, 3, 16) # pour 16 ml of green paint into beaker 3
POUR(2, 3, 31) # pour 31 ml of blue paint into beaker 3
DONE(3) # the desired mixture is created in beaker 3
--- PLAN END ---
"""

example2 = """
Beaker 0: Color: RGB(255, 0, 0), amount: 100ml
Beaker 1: Color: RGB(0, 255, 0), amount: 100ml
Beaker 2: Color: RGB(0, 0, 255), amount: 100ml
Beaker 3: Color: RGB(0, 0, 0), amount: 0ml
Target beaker: RGB(76, 107, 43), amount: 120ml
--- REASONING START ---
In this environment, we have primary colors red, green, and blue in equal amounts, and we need to create a dark green shade in the target beaker. The target color has the following RGB values: R=76, G=107, B=43. Since we're working with a subtractive color mixing model, we need to mix the colors in such a way that we achieve this specific dark green shade.

To get the color right, we need to mix more green than red and blue because the green value is the highest in the target color. The red value is slightly higher than the blue value, so we'll need a bit more red than blue. Since we need 120ml of paint in total, we can try to maintain the ratio of the RGB values of the target color in our mixture.

The ratio of the RGB values in the target color is approximately 76:107:43. To simplify the ratio, we can divide each number by the smallest value, which is 43, giving us approximately 1.76:2.49:1. We can round these to the nearest whole numbers to make the mixing easier, resulting in a ratio of 2:2.5:1. To scale this ratio up to 120ml, we can multiply each part of the ratio by a factor that will give us a total of 120ml. The sum of the parts of the ratio is 5.5, and 120 divided by 5.5 gives us approximately 21.8. Multiplying the simplified ratio by this factor gives us amounts of approximately 44ml red, 55ml green, and 22ml blue.

However, we need to adjust these amounts slightly to ensure we have whole numbers and that the total amount is exactly 120ml. We can round the red to 44ml, the green to 54ml, and the blue to 22ml, which gives us a total of 120ml.

Now, let's plan the sequence of actions to create the target beaker.
--- REASONING END ---

--- PLAN START ---
POUR(1, 3, 54) # pour 54 ml of green paint into the empty beaker 3
POUR(0, 3, 44) # pour 44 ml of red paint into beaker 3
POUR(2, 3, 22) # pour 22 ml of blue paint into beaker 3
DONE(3) # the desired mixture was created in beaker 3
--- PLAN END ---
"""

green = """
Beaker 0: Color: RGB(255, 0, 0), amount: 100ml
Beaker 1: Color: RGB(0, 255, 0), amount: 100ml
Beaker 2: Color: RGB(0, 0, 255), amount: 100ml
Beaker 3: Color: RGB(0, 0, 0), amount: 0ml
Target beaker: RGB(76, 107, 43), amount: 120ml
--- REASONING START ---
In this environment, we have primary colors red, green, and blue in equal amounts, and we need to create a dark green shade in the target beaker. The target color has the following RGB values: R=76, G=107, B=43. Since we're working with a subtractive color mixing model, we need to mix the colors in such a way that we achieve this specific dark green shade.

To get the color right, we need to mix more green than red and blue because the green value is the highest in the target color. The red value is slightly higher than the blue value, so we'll need a bit more red than blue. Since we need 120ml of paint in total, we can try to maintain the ratio of the RGB values of the target color in our mixture.

The ratio of the RGB values in the target color is approximately 76:107:43. To simplify the ratio, we can divide each number by the smallest value, which is 43, giving us approximately 1.76:2.49:1. We can round these to the nearest whole numbers to make the mixing easier, resulting in a ratio of 2:2.5:1. To scale this ratio up to 120ml, we can multiply each part of the ratio by a factor that will give us a total of 120ml. The sum of the parts of the ratio is 5.5, and 120 divided by 5.5 gives us approximately 21.8. Multiplying the simplified ratio by this factor gives us amounts of approximately 44ml red, 55ml green, and 22ml blue.

However, we need to adjust these amounts slightly to ensure we have whole numbers and that the total amount is exactly 120ml. We can round the red to 44ml, the green to 54ml, and the blue to 22ml, which gives us a total of 120ml.

Now, let's plan the sequence of actions to create the target beaker.
--- REASONING END ---

--- PLAN START ---
POUR(1, 3, 54) # pour 54 ml of green paint into the empty beaker 3
POUR(0, 3, 44) # pour 44 ml of red paint into beaker 3
POUR(2, 3, 22) # pour 22 ml of blue paint into beaker 3
DONE(3) # the desired mixture was created in beaker 3
--- PLAN END ---
"""


planner_template = PromptTemplate(
    input_variables=['state', 'prev_reasoning', 'prev_plan', 'feedback'],
    template=planner_instructions
)

critic_template = PromptTemplate(
    input_variables=['initial_state', 'plan_reasoning', 'plan', 'trajectory'],
    template=critic_instructions
)

