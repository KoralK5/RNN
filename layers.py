import numpy as np
import activation

# one single GRU layer
class GRU:
    def __init__(self):
        return

    def forwprop(self):
        return

    def backprop(self):
        return

# one single GRU layer
class LSTM:
    def __init__(self, prev_cell, prev_hidden, cur_input):
        self.prev_cell = prev_cell
        self.prev_hidden = prev_hidden
        self.cur_input = cur_input

        self.new_cell = None
        self.new_hidden = None

    def forwprop(self):
        prev_hidden = self.prev_hidden
        cur_input = self.cur_input

        concat_val = np.concatenate([prev_hidden, cur_input])

        # cell state calculations
        forget_out = activation.sigmoid(concat_val) # forget gate
        candidate_out = activation.tanh(concat_val) # candidate output
        input_out = np.multiply(forget_out, candidate_out) # input gate
        new_cell = np.multiply(input_out, forget_out) + input_out

        # hidden state calculations
        new_hidden = np.multiply(forget_out, activation.sigmoid(new_cell))

        # hidden state can be used for predictions
        return new_cell, new_hidden

    def backprop(self):
        return

if __name__=='__main__':
    param_size = 2
    cell_state = np.random.uniform(low=0, high=1, size=param_size)
    hidden_state = np.random.uniform(low=0, high=1, size=param_size)
    inputs = [np.array([0, 0]), np.array([0, 1]), np.array([1, 0]), np.array([1, 1])]

    layer_count = 4
    model = LSTM(cell_state, hidden_state, inputs[0])
    for i in range(layer_count):
        model.cur_input = inputs[i]
        model.cell_state, model.hidden_state = model.forwprop()

    print(cell_state)
    print(hidden_state)

