# What are we optimising for?
Q values for an output neuron given a current image
    state = the image
    action = the output neuron

immediate_reward = 1 if output neuron index == label (else 0)

# Policy
Action is determined by epsilon greedy policy (already implemented)

# TD error
td_error = immediate_reward + discount_factor * max_q_in_next_state - current_q

This 

# Update Q value of previous state and action just taken
q_value = current_q + learning_rate * td_error