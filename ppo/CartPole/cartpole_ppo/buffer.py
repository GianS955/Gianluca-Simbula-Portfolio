import numpy as np

class RolloutBuffer:
    def __init__ (self, num_states, num_actions,layers):
        self.num_states = num_states
        self.num_actions = num_actions
        self.buffer = {}
        self.current_index = 0
    
    def __len__(self):
        return len(list(self.buffer.keys()))
    
    def append(self, state, action, reward, state_value, log_prob, action_probs, is_terminal, actor_cache, critic_cache,index = None):
        if index is None:
            current_index = self.current_index
            self.current_index += 1
        else:
            current_index = index

        self.buffer[current_index] = {}
        self.buffer[current_index]['state'] = state.reshape(1,-1)
        self.buffer[current_index]['one_hot'] = action.reshape(1,-1)
        self.buffer[current_index]['reward'] = np.array([[reward]])
        self.buffer[current_index]['state_value'] = np.array([[float(state_value)]])
        self.buffer[current_index]['log_prob'] = np.array([[log_prob]])
        self.buffer[current_index]['action_prob'] = action_probs.reshape(1,-1)
        self.buffer[current_index]['is_terminal'] = np.array([[is_terminal]])
        self.buffer[current_index]['actor_cache'] = {}
        for i in list(actor_cache.keys()):
            self.buffer[current_index]['actor_cache'][i] = {}
            self.buffer[current_index]['actor_cache'][i]['input'] = np.reshape(actor_cache[i]['input'], (1, -1))
            self.buffer[current_index]['actor_cache'][i]['output'] = np.reshape(actor_cache[i]['output'], (1, -1))
        self.buffer[current_index]['critic_cache'] = {}
        for i in list(critic_cache.keys()):
            self.buffer[current_index]['critic_cache'][i] = {}
            self.buffer[current_index]['critic_cache'][i]['input'] = np.reshape(critic_cache[i]['input'], (1, -1))
            self.buffer[current_index]['critic_cache'][i]['output'] = np.reshape(critic_cache[i]['output'], (1, -1))

    def empty_buffer(self):
        self.current_index = 0
        self.buffer = {}

    
    def get_actor_cache(self, layer, cache_type, indices = None):
        if indices is None:
            return np.vstack([self.buffer[i]['actor_cache'][layer][cache_type] for i in range(len(self))])
        else:
            return np.vstack([self.buffer[i]['actor_cache'][layer][cache_type] for i in indices])

    def get_critic_cache(self, layer, cache_type, indices = None):
        if indices is None:
            return np.vstack([self.buffer[i]['critic_cache'][layer][cache_type] for i in range(len(self))])
        else:
            return np.vstack([self.buffer[i]['critic_cache'][layer][cache_type] for i in indices])

    def get(self, key, indices = None):
        if indices is None:
            return np.vstack([self.buffer[i][key] for i in range(len(self))])
        elif isinstance(indices, (int, np.integer)):
            return self.buffer[indices][key]
        else:
            return np.vstack([self.buffer[i][key] for i in indices])

    def set(self,values, key, indices = None):
        
        if indices is None:
            for i in range(values.shape[0]):
                self.buffer[i][key] = values[i]
        else:
            for i in range(len(indices)):
                self.buffer[indices[i]][key] = values[i]

    def get_minibatch(self, batch_size):
        N = len(self)
        indices = np.random.permutation(N)

        return indices.tolist()[:batch_size]
