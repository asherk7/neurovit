"""
multilayer perceptron of the transformer

according to the paper:
we normalize the input, then we put it through two linear layers separated by a GELU activation
each linear layer is followed by a dropout layer
structure: normalize --> linear --> GELU --> dropout (0.1 in table 3) --> linear --> dropout
"""