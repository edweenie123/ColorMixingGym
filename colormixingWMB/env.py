from typing import Tuple, Optional, List

class SubtractiveModel:
    @staticmethod
    def _rgb_to_cmy(rgb: Tuple[int, int, int]) -> Tuple[int, int, int]:
        return tuple(255 - value for value in rgb)

    @staticmethod
    def _cmy_to_rgb(cmy: Tuple[int, int, int]) -> Tuple[int, int, int]:
        return tuple(255 - value for value in cmy)

    @staticmethod
    def mix_colors(r1, g1, b1, a1, r2, g2, b2, a2) -> Tuple[int, int, int]:
        cmy1 = SubtractiveModel._rgb_to_cmy((r1, g1, b1))
        cmy2 = SubtractiveModel._rgb_to_cmy((r2, g2, b2))
        # cmy2 = SubtractiveModel._rgb_to_cmy(paint2.color)
        total_amount = a1 + a2
        if a1 == 0:
            return (r2, g2, b2)
        if a2 == 0:
            return (r1, g1, b1)

        mixed_cmy = tuple(int((cmy1[i] * a1 + cmy2[i] * a2) / total_amount) for i in range(3))
        return SubtractiveModel._cmy_to_rgb(mixed_cmy)



class ColorMixing:
    
    def state_transition(self, state, action):
        """
        This is the GROUND TRUTH state transition function!

        Applies an action to the current state and returns the new state.

        :param state: A set of predicates representing the current state.
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

            transfer_amt = min(amt, src_amt)
            new_src_amt =  src_amt - transfer_amt
            new_tgt_amt = tgt_amt + transfer_amt
            new_tgt_color = SubtractiveModel.mix_colors(src_r, src_g, src_b, transfer_amt, tgt_r, tgt_g, tgt_b, tgt_amt)
            new_tgt_r, new_tgt_g, new_tgt_b = new_tgt_color
            
            new_src_contains = f"contains {src_idx} {src_r} {src_g} {src_b} {new_src_amt}"
            new_tgt_contains = f"contains {tgt_idx} {new_tgt_r} {new_tgt_g} {new_tgt_b} {new_tgt_amt}"

            new_state.discard(src_contains)
            new_state.discard(tgt_contains)

            new_state.add(new_src_contains)
            new_state.add(new_tgt_contains)
            
        return new_state