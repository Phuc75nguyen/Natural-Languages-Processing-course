import numpy as np

# Define the components of the HMM
states = ['Noun', 'Verb', 'Determiner']  # POS tags
observations = ['Janet', 'will', 'back', 'the', 'bill']  # Sentence
start_probability = {'Noun': 0.5, 'Verb': 0.1, 'Determiner': 0.4}  # Initial probabilities
transition_probability = {
    'Noun': {'Noun': 0.1, 'Verb': 0.6, 'Determiner': 0.3},
    'Verb': {'Noun': 0.3, 'Verb': 0.2, 'Determiner': 0.5},
    'Determiner': {'Noun': 0.8, 'Verb': 0.1, 'Determiner': 0.1},
}
emission_probability = {
    'Noun': {'Janet': 0.4, 'will': 0.1, 'back': 0.1, 'the': 0.05, 'bill': 0.35},
    'Verb': {'Janet': 0.05, 'will': 0.6, 'back': 0.3, 'the': 0.05, 'bill': 0.0},
    'Determiner': {'Janet': 0.0, 'will': 0.0, 'back': 0.0, 'the': 0.8, 'bill': 0.2},
}

# Viterbi algorithm implementation
def viterbi(obs, states, start_p, trans_p, emit_p):
    T = len(obs)
    N = len(states)
    
    viterbi_matrix = np.zeros((N, T))
    backpointer = np.zeros((N, T), dtype=int)
    
    # State indexing
    state_index = {state: idx for idx, state in enumerate(states)}
    
    # Initialization step
    for s in states:
        idx = state_index[s]
        viterbi_matrix[idx, 0] = start_p[s] * emit_p[s].get(obs[0], 0)
        backpointer[idx, 0] = 0

    # Recursion step
    for t in range(1, T):
        for s in states:
            current_idx = state_index[s]
            max_prob = -1
            max_state = 0
            for s_prev in states:
                prev_idx = state_index[s_prev]
                prob = viterbi_matrix[prev_idx, t - 1] * trans_p[s_prev][s] * emit_p[s].get(obs[t], 0)
                if prob > max_prob:
                    max_prob = prob
                    max_state = prev_idx
            viterbi_matrix[current_idx, t] = max_prob
            backpointer[current_idx, t] = max_state
    
    # Termination step
    best_path_prob = -1
    best_last_state = 0
    for s in states:
        idx = state_index[s]
        if viterbi_matrix[idx, T - 1] > best_path_prob:
            best_path_prob = viterbi_matrix[idx, T - 1]
            best_last_state = idx

    # Backtracking to find the best path
    best_path = []
    current_state = best_last_state
    for t in range(T - 1, -1, -1):
        best_path.insert(0, states[current_state])
        current_state = backpointer[current_state, t]

    return best_path, best_path_prob

# Run the algorithm
best_path, best_path_prob = viterbi(observations, states, start_probability, transition_probability, emission_probability)

# Output the result
print("Best Path (POS Tags):", best_path)
print("Probability of Best Path:", best_path_prob)
