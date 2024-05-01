import math

def beaker_similarity(beaker1, beaker2):
    """
    Computes the similarity between two beakers based on RGB color and paint volume.
    Ignores color similarity if the amount in either beaker is zero.

    :param beaker1: Contains string for the first beaker (e.g., "contains idx r g b amount").
    :param beaker2: Contains string for the second beaker.
    :return: Similarity score between 0 and 1.
    """
    _, idx1, r1, g1, b1, amt1 = beaker1.split()
    _, idx2, r2, g2, b2, amt2 = beaker2.split()

    # Convert strings to integers
    r1, g1, b1, amt1 = int(r1), int(g1), int(b1), int(amt1)
    r2, g2, b2, amt2 = int(r2), int(g2), int(b2), int(amt2)

    # Normalize amounts for comparison
    max_amount = 200  # Arbitrary max volume
    normalized_amt1 = amt1 / max_amount
    normalized_amt2 = amt2 / max_amount

    # Compute volume similarity directly
    volume_similarity = 1 - abs(normalized_amt1 - normalized_amt2)

    # Weight factors for color and volume can be adjusted based on the importance of each aspect
    color_weight = 0.9
    volume_weight = 0.1

    # Only compute color similarity if both amounts are non-zero
    if amt1 > 0 and amt2 > 0:
        # Calculate color distance
        color_distance = math.sqrt((r1 - r2)**2 + (g1 - g2)**2 + (b1 - b2)**2)
        max_color_distance = math.sqrt(3 * 255**2)  # Maximum possible color distance
        color_similarity = 1 - (color_distance / max_color_distance)
    else:
        # If either amount is zero, color similarity doesn't matter
        color_similarity = 0
        volume_weight = 1


    # Compute weighted average of similarities
    total_similarity = (color_weight * color_similarity + volume_weight * volume_similarity)

    return total_similarity

def state_similarity(state1, state2):
    """
    Computes the average similarity between two states, each a set of beaker descriptions.
    Beakers are matched based on their identifiers before calculating similarities.

    :param state1: Set of contains strings for the first state.
    :param state2: Set of contains strings for the second state.
    :return: Average similarity score between 0 and 1.
    """
    # Parse states into dictionaries keyed by beaker index
    state1_dict = {beaker.split()[1]: beaker for beaker in state1}
    state2_dict = {beaker.split()[1]: beaker for beaker in state2}

    similarities = []
    # Only compare beakers that exist in both states
    common_indices = state1_dict.keys() & state2_dict.keys()
    for idx in common_indices:
        beaker1 = state1_dict[idx]
        beaker2 = state2_dict[idx]
        similarity = beaker_similarity(beaker1, beaker2)
        similarities.append(similarity)

    # Calculate average similarity, ensure there are common beakers to compare
    average_similarity = sum(similarities) / len(similarities) if similarities else 0
    return average_similarity


def rescaled_state_similarity(prev_state, real_next_state, pred_next_state):
    """
    Rescales state simliarity so that 0.5 is the same as doing nothing to prev_state
    """
    
    do_nothing_sim = state_similarity(prev_state, real_next_state)
    real_pred_sim = state_similarity(real_next_state, pred_next_state)
    
    rescaled_similarity = 0.5 + (real_pred_sim - do_nothing_sim) / (2 * max(abs(real_pred_sim - do_nothing_sim), 1 - do_nothing_sim))
    
    return rescaled_similarity

if __name__ == "__main__":    
    prev_state = {"contains 1 251 0 1 92", "contains 2 0 255 0 100", "contains 3 16 0 149 145", "contains 4 232 232 232 110", "contains 5 45 0 110 8", "contains 6 0 0 67 45"}
    pred_next_state = {"contains 1 251 0 1 92", "contains 2 0 255 0 100", "contains 3 16 0 149 145", "contains 4 232 232 232 110", "contains 5 45 0 110 8", "contains 6 0 0 67 45"}
    actual_next_state = {"contains 1 251 0 1 92", "contains 2 0 255 0 90", "contains 3 16 0 149 145", "contains 4 232 232 232 110", "contains 5 20 142 49 18", "contains 6 0 0 67 45"}
    print(rescaled_state_similarity(prev_state, actual_next_state, pred_next_state))


    


