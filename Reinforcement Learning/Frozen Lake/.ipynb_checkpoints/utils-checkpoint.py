import numpy as np
import numpy.random as random
import matplotlib.pyplot as plt

class Action:
    def __init__(self,next_tile, current_tile):
        action_to_vector = {
            (0,1) : "L",
            (0,-1): "R",
            (1,0) : "D",
            (-1,0): "U"
        }

        direction = (next_tile.i - current_tile.i, next_tile.j - current_tile.j)
        if (direction in action_to_vector):
            self.Arrow = direction
            self.Direction = action_to_vector[direction]
        else:
            print("ERRORE",direction)
            
class Tile:
    def __init__(self,i=0,j=0,t=None):
        self.i = i
        self.j = j
        types = ["F", "H","S", "E"]
        if t not in types:
            raise Exception(f'{t} tile type not valid')
        self.Type = t
        self._value = 0
        self._actions = []

        match self.Type:
            case "F":
                self._reward = -1
            case "H":
                self._reward = -30
            case "S":
                self._reward = -1
            case "E":
                self._reward = 100

    @property
    def reward(self):
        return self._reward

    @reward.setter
    def reward(self, r):
        self._reward = r

    @property
    def actions(self):
        return self._actions
    @actions.setter
    def actions(self, a):
        self._actions = a

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, v):
        self._value = v

class Lake:
    def __init__(self):
        self.Grid = []

    def ImportCSV(self,file):
        import csv
        i = 0
        j = 0
        with open(file, newline = '') as csv_file:
            rdr = csv.reader(csv_file, delimiter = ',')
            for row in rdr:
                r = []
                for tile in row:
                    r.append(Tile(i, j,tile.strip()))
                    j +=1
                self.Grid.append(r)
                j = 0
                i+=1

        self.shape= [len(self.Grid),len(self.Grid[0])]
                    
        
    def GetNeighbours(self, tile):
        neighbours = []

        # vertical neighbors
        for i in [tile.i - 1, tile.i + 1]:
            if 0 <= i < self.shape[0]:
                neighbours.append(self.Grid[i][tile.j])

        # horizontal neighbors
        for j in [tile.j - 1, tile.j + 1]:
            if 0 <= j < self.shape[1]:
                neighbours.append(self.Grid[tile.i][j])

        return neighbours
                    
    def Initialize(self):
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                current_tile = self.Grid[i][j]
                if (current_tile.Type in ["F","S"]):
                    self.Grid[i][j].value = 0
                else:
                    self.Grid[i][j].value = current_tile.reward        

    def PolicyIteration(self, max_iterations, treshold, gamma):
        import time
        t = 0
        while (True):
            start = time.perf_counter()    
            # policy-evaluation:
            deltas = np.zeros(self.shape)
            V = np.zeros(self.shape)
            for i in range(self.shape[0]):
                for j in range(self.shape[1]):
                    current_tile = self.Grid[i][j]
                    if (current_tile.Type in ['F','S']):
                        neighbours = self.GetNeighbours(current_tile)
                        values = [n.value for n in neighbours]
                        new_value = np.sum(values)/len(neighbours)
                        deltas[i,j] = abs(new_value - current_tile.value)
                        V[i,j] = new_value

            for i in range(self.shape[0]):
                for j in range(self.shape[1]):
                    current_tile = self.Grid[i][j]
                    if (current_tile.Type in ['F','S']):
                        self.Grid[i][j].value = V[i,j]

            t+=1
            d = np.max(deltas)
            if (d<=delta):
                print(f'Convergence reached')
                break
            if (t >= iterations):
                print(f'Max number of iterations reached.')        
            break

        # policy-improvement (Control)
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                current_tile = self.Grid[i][j]
                if (current_tile.Type in ["F","S"]):
                    neighbours = self.GetNeighbours(current_tile)
                    values = [neighbour.reward + gamma*neighbour.value for neighbour in neighbours]
                    max_indexes = np.argwhere(values == np.max(values)).flatten().tolist()
                    for index in max_indexes:
                        self.Grid[i][j].actions.append(Action(neighbours[index],current_tile))

        end = time.perf_counter()
        time_pi = end-start
        print(f"Time passed {time_pi} seconds. {t} iterations")
        
    def ValueIteration(self, max_iterations, treshold, gamma, env = 'deterministic'):
        import time
        if (env == 'stochastic'):
            hole_reward = Tile(t="H").reward
        t = 0
        while (True):
            start = time.perf_counter()
            deltas = np.zeros([self.shape[0],self.shape[1]])
            V = np.zeros(self.shape)
            for i in range(self.shape[0]):
                for j in range(self.shape[1]):
                    current_tile = self.Grid[i][j]
                    if (current_tile.Type in ["F","S"]):
                        neighbours = self.GetNeighbours(current_tile)
                        values = []
                        for neighbour in neighbours:
                            if (neighbour.Type in ['F','S']):
                                if (env == 'deterministic'): 
                                    values.append(neighbour.reward + gamma*neighbour.value)
                                elif (env == 'stochastic'):
                                    next_neighbours = self.GetNeighbours(neighbour)
                                    p_safe = sum([n.Type in ["F","S"] for n in next_neighbours])/4
                                    values.append(p_safe*(neighbour.reward + gamma*neighbour.value) + (1-p_safe)*(hole_reward))
                                else:
                                    raise Exception(f'{ennvironment} environment not recognized.')
                                
                            else:
                                values.append(neighbour.reward)
                                        
                        n_max = np.argmax(values)
                        deltas[i,j] = abs(np.max(values) - self.Grid[i][j].value)
                        V[i,j] = np.max(values)
                        
            for i in range(self.shape[0]):
                for j in range(self.shape[1]):
                    current_tile = self.Grid[i][j]
                    if (current_tile.Type in ["F","S"]):
                        self.Grid[i][j].value = V[i,j]
            
            t+=1
            d = np.max(deltas)
            if (d<=treshold):
                print(f'Convergence reached')
                break
            if (t >= max_iterations):
                print(f'Max number of iterations reached.')        
                break
        
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                current_tile = self.Grid[i][j]
                if (current_tile.Type in ["F","S"]):
                    neighbours = self.GetNeighbours(current_tile)
                    values = [n.reward + gamma*n.value for n in neighbours]
                    max_indexes = np.argwhere(values == np.max(values)).flatten().tolist()
                    for index in max_indexes:
                        self.Grid[i][j].actions.append(Action(neighbours[index],current_tile))
            
        end = time.perf_counter()
        time_vi = end-start
        print(f"Time passed {time_vi} seconds. {t} iterations")

        
    def Print(self):
        for i in range(self.shape[0]):
            res = []
            for j in range(self.shape[1]):
                res.append(self.Grid[i][j].Type)
            print(','.join(res))

    def Plot(self):
        import matplotlib.pyplot as plt
        plt, ax = plt.subplots()
        heatmap = np.empty(self.shape)
        annotations = []
        for i in range(self.shape[0]):
            row = []
            for j in range(self.shape[1]):
                heatmap[i,j] = self.Grid[i][j].reward
                row.append(self.Grid[i][j].Type)
            annotations.append(row)
        
        im = ax.imshow(heatmap)

        for i in range(len(annotations)):
            for j in range(len(annotations[i])):
                text = ax.text(j, i, annotations[i][j], ha="center", va="center", color="k")
        

    def PrintValues(self):
        import matplotlib.pyplot as plt
        plt, ax = plt.subplots()
        heatmap = np.empty(self.shape)
        values = []
        for i in range(self.shape[0]):
            row = []
            for j in range(self.shape[1]):
                heatmap[i,j] = self.Grid[i][j].value
                row.append(round(self.Grid[i][j].value,1))
            values.append(row)
        
        im = ax.imshow(heatmap)

        for i in range(len(values)):
            for j in range(len(values[i])):
                text = ax.text(j, i, values[i][j], ha="center", va="center", color="k")
        

    def PrintActions(self):
        import matplotlib.pyplot as plt
        plt, ax = plt.subplots()
        heatmap = np.empty(self.shape)
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                heatmap[i,j] = self.Grid[i][j].reward
        
        im = ax.imshow(heatmap)
        for action_set in range(4):
            U = np.zeros(self.shape)
            V = np.zeros(self.shape)
            for i in range(self.shape[0]):
                for j in range(self.shape[1]):
                    try:
                        a = self.Grid[i][j].actions[action_set]
                    except:
                        continue
                    
                    dx, dy = a.Arrow
                    U[i, j] = dy
                    V[i, j] = -dx

            ax.quiver(
                np.arange(self.shape[1]),
                np.arange(self.shape[0]),
                U, V,
                units='xy', scale=2, scale_units='xy',
                color="black", width=0.05
            )


class Environment:
    def __init__(self,shape):
        self.Lake = Lake(shape)
        self.A = ["L","U","R","D"]