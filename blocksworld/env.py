class BlocksWorld:
    
    def state_transition(self, state, action):
        """
        This is the GROUND TRUTH state transition function!

        Applies an action to the current state and returns the new state.

        :param state: A set of predicates representing the current state.
        :param action: A string representing the action to be applied.
        :return: The new state as a set of predicates.
        """
        # Split action into words to extract action type and parameters
        words = action.split()
        action_type = words[0]
        params = words[1:]

        # Copy the current state to avoid mutating the original
        new_state = set(state)

        if action_type == "pick-up":
            x = params[0]
            if {"clear " + x, "ontable " + x, "handempty"} <= state:
                new_state.discard("ontable " + x)
                new_state.discard("clear " + x)
                new_state.discard("handempty")
                new_state.add("holding " + x)

        elif action_type == "put-down":
            x = params[0]
            if {"holding " + x} <= state:
                new_state.add("ontable " + x)
                new_state.add("clear " + x)
                new_state.add("handempty")
                new_state.discard("holding " + x)

        elif action_type == "stack":
            x, y = params
            if {"holding " + x, "clear " + y} <= state:
                new_state.discard("holding " + x)
                new_state.discard("clear " + y)
                new_state.add("clear " + x)
                new_state.add("handempty")
                new_state.add("on " + x + " " + y)

        elif action_type == "unstack":
            x, y = params
            if {"on " + x + " " + y, "clear " + x, "handempty"} <= state:
                new_state.add("holding " + x)
                new_state.add("clear " + y)
                new_state.discard("clear " + x)
                new_state.discard("handempty")
                new_state.discard("on " + x + " " + y)

        return new_state