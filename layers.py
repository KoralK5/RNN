import numpy as np
import activation

# one single GRU layer
class GRU:
    def __init__(self, h_dim, h_new_dim, x_dim):
        # initialize the following weights
        # W_r, W_z, W, U_r, U_z, U, r_t, z_t, h_not_t

    def forwprop():
        # do stuff

    def backprop():
        # do stuff

# one single GRU layer
class LSTM:
    def __init__(self, prev_cell, prev_hidden, cur_input):
        self.prev_cell = prev_cell
        self.prev_hidden = prev_hidden
        self.cur_input = cur_input

        self.new_cell = np.array()
        self.new_hidden = np.array()

    def forwprop():
        prev_hidden = self.prev_hidden
        cur_input = self.cur_input

        concat_val = np.concat(prev_hidden, cur_input)

        # cell state calculations
        forget_out = activation.sigmoid(concat_val) # forget gate
        candidate_out = activation.tanh(concat_val) # candidate output
        input_out = np.multiply(forget_out, candidate_out) # input gate
        self.new_cell = np.concat(np.multiply(self.old_cell, self.forget_out), input_out)

        # hidden state calculations
        self.new_hidden = np.multiply(self.forget_out, np.sigmoid(self.new_cell))

    def backprop():
        # do stuff

