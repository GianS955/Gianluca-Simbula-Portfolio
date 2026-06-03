import numpy as np

class Optimizer:
    def __init__(self, optimization_info):
        self.learning_rate = optimization_info.get('learning_rate', 0.001)
        self.beta_1 = optimization_info.get('beta_1', 0.9)
        self.beta_2 = optimization_info.get('beta_2', 0.999)
        self.m = {}
        self.v = {}

    def optimize_gradient(self, gradient, layer, parameter_type):
        if layer not in self.m:
            self.m[layer] = {}
            
        if layer not in self.v:
            self.v[layer] = {}

        if parameter_type not in self.m[layer]:
            self.m[layer][parameter_type] = np.zeros_like(gradient)
        if parameter_type not in self.v[layer]:
            self.v[layer][parameter_type] = np.zeros_like(gradient)
            
        self.m[layer][parameter_type] = self.beta_1 * self.m[layer][parameter_type] + (1 - self.beta_1) * gradient
        self.v[layer][parameter_type] = self.beta_2 * self.v[layer][parameter_type] + (1 - self.beta_2) * (gradient ** 2)
        m_hat = self.m[layer][parameter_type] / (1 - self.beta_1)
        v_hat = self.v[layer][parameter_type] / (1 - self.beta_2)
        return - self.learning_rate * m_hat / (np.sqrt(v_hat) + 1e-8)
    