import numpy as np
from optimizer import Optimizer

class NeuralNetwork:
    def __init__(self, network_info, optimizer_info,name):
        self.w = {}
        self.b = {}
        self.name = name
        self.layers = [network_info['state_dimensions']] + network_info['hidden_layer_dimensions'] + [network_info.get('action_dimensions', 1)]
        self.network = {}
        for i in range(len(self.layers)):
            if i == 0:
                self.network[i] = {"num_neurons": self.layers[i], 'activation': None}
            else:
                self.network[i] = {"num_neurons": self.layers[i], 'activation': network_info['activation']}
        self.optimizer = Optimizer(optimizer_info)

    def initialize(self):
        for i in range(len(self.layers)-1):
            limit = np.sqrt(1/self.layers[i])
            self.w[i] = np.random.uniform(-limit, limit, size=(self.layers[i],self.layers[i+1]))
            self.b[i] = np.zeros(shape=(1,self.layers[i+1]))
    
    def forward_pass(self,states):
        value = np.reshape(states, (1, -1))
        cache = {}
        for layer in self.w.keys():
            cache[layer] = {"input": value, "output": None}
            psi = value @ self.w[layer] + self.b[layer]
            if layer != list(self.w.keys())[-1]:
                if self.network[layer]['activation'] == 'tanh':
                    value = np.tanh(psi)
                elif self.network[layer]['activation'] == 'relu':
                    value = np.maximum(psi,0)
                elif self.network[layer]['activation'] == None:
                    value = psi
            else:
                value = psi
            cache[layer]['output'] = value
        return value, cache
    
    def activation_derivative(self, value, layer):
        if self.network[layer]['activation'] == 'tanh':
            return 1 - value**2
        elif self.network[layer]['activation'] == 'relu':
            return (value > 0).astype(float)
        elif self.network[layer]['activation'] == None:
            return np.ones_like(value)
        
    def update_parameters(self, w_gradients, b_gradients):
        if self.name == 'actor':
             for layer in self.w.keys():
                self.w[layer] = self.w[layer] + self.optimizer.optimize_gradient(w_gradients[layer], layer, 'w')
                self.b[layer] = self.b[layer] + self.optimizer.optimize_gradient(b_gradients[layer], layer, 'b')
        else:
            for layer in self.w.keys():
                self.w[layer] = self.w[layer] - self.optimizer.optimize_gradient(w_gradients[layer], layer, 'w')
                self.b[layer] = self.b[layer] - self.optimizer.optimize_gradient(b_gradients[layer], layer, 'b')