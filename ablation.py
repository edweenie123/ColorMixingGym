from agent import *


# remove critic
class UnreactivePlanner(LLMAgent):
    def __init__(self):
        llm = ChatOpenAI(
            temperature=0,
            # model_name='gpt-3.5-turbo-1106'
            model_name='gpt-4-1106-preview'
        )

        self.planner = LLMChain(llm=llm, prompt=planner_template_no_feedback)
        # self.critic = LLMChain(llm=llm, prompt=critic_template)
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
        print(f'input to planner: {obs} \n {self.plan_reasoning} \n {self.plan} \n {self.trajectory}')
        planner_response = self.planner.run(
            state=obs,
            prev_reasoning=self.plan_reasoning,
            prev_plan=self.plan,
            trajectory=self.trajectory
        )
        
        # reset trajectory
        self.trajectory = ''
        


        print(f'planner_response={planner_response}')
        self.planner_response = planner_response
        self.plan_reasoning = extract_section(planner_response, 'REASONING')
        self.plan = extract_section(planner_response, 'PLAN') 

        self.instructions = extract_instructions(planner_response)
        self.inst_idx = 0
        
        return find_next_execute()
    
    def end_episode_reflection(self):

        self.instructions = None


class UnreactivePlannerCritic_no_few_shot(LLMAgent):
    def __init__(self):
        llm = ChatOpenAI(
            temperature=0,
            # model_name='gpt-3.5-turbo-1106'
            model_name='gpt-4-1106-preview'
        )

        self.planner = LLMChain(llm=llm, prompt=planner_template_no_few_shot)
        self.critic = LLMChain(llm=llm, prompt=critic_template_no_few_shot)
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


class UnreactivePlannerCritic_no_cot(LLMAgent):
    def __init__(self):
        llm = ChatOpenAI(
            temperature=0,
            # model_name='gpt-3.5-turbo-1106'
            model_name='gpt-4-1106-preview'
        )

        self.planner = LLMChain(llm=llm, prompt=planner_template_no_cot)
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
        # self.plan_reasoning = '' # stores the current plan_reasoning
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
            # prev_reasoning=self.plan_reasoning,
            prev_plan=self.plan,
            trajectory=self.trajectory
        )
        
        print(f'planner_response={planner_response}')

        self.planner_response = planner_response
        # self.plan_reasoning = extract_section(planner_response, 'REASONING')
        self.plan = extract_section(planner_response, 'PLAN') 
        self.trajectory = ''

        self.instructions = extract_instructions(planner_response)
        self.inst_idx = 0
        
        return find_next_execute()
    
    def end_episode_reflection(self):
        self.instructions = None

if __name__ == '__main__':
    for file in os.listdir('data/level1/'):
        print(f'Starting test for {file}')
        file_path = os.path.join('data/level1/', file)
        
        agent = UnreactivePlannerCritic_no_cot()
        iterative_control_loop(file_path, agent, num_iter=3, render=False, log_path=f'results/ablation/no_cot/{file}.csv')




