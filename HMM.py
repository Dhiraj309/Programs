import numpy as np

# Define the states, observations, transition, and emission probabilities
transition_probability = {
    'Sunny': {'Sunny': 0.8, 'Rainy': 0.2},
    'Rainy': {'Sunny': 0.5, 'Rainy': 0.5}
}

emission_probability = {
    'Sunny': {'Walk': 0.7, 'Shop': 0.2, 'Clean': 0.1},
    'Rainy': {'Walk': 0.2, 'Shop': 0.4, 'Clean': 0.4}
}

# Viterbi Algorithm for predicting the most probable state sequence
def viterbi(obs, states, start_p, trans_p, emit_p):
    V = [{}]
    path = {}
    
    # Initialize base cases (t == 0)
    for y in states:
        V[0][y] = start_p[y] * emit_p[y][obs[0]]
        path[y] = [y]
    
    # Run Viterbi for t > 0
    for t in range(1, len(obs)):
        V.append({})
        newpath = {}
        
        for y in states:
            (prob, state) = max((V[t-1][y0] * trans_p[y0][y] * emit_p[y][obs[t]], y0) for y0 in states)
            V[t][y] = prob
            newpath[y] = path[state] + [y]
        
        path = newpath
    
    # Find the final highest probability
    n = len(obs) - 1
    (prob, state) = max((V[n][y], y) for y in states)
    return (prob, path[state])

# Define states, observations, and start probabilities
states = ['Sunny', 'Rainy']
observations = ['Walk', 'Shop', 'Clean', 'Walk']
start_probability = {'Sunny': 0.7, 'Rainy': 0.3}

# Test the model
obs_sequence = ['Walk', 'Shop', 'Clean', 'Walk']
probability, state_sequence = viterbi(obs_sequence, states, start_probability, transition_probability, emission_probability)

# Output the result
print("Observation Sequence:", obs_sequence)
print("Most Probable State Sequence:", state_sequence)
print("Probability of the Sequence:", probability)
