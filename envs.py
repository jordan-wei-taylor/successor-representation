from   matplotlib import pyplot as plt
from   typing     import Union
import numpy      as np

one_room = """
OOOOOOOOOO
OOOOOOOOOO
OOOOOOOOOO
OOOOOOOOOO
OOOOOOOOOO
OOOOOOOOOO
OOOOOOOOOO
OOOOOOOOOO
OOOOOOOOOO
OOOOOOOOOO
"""

# 1
i_maze = """
O             O
OOOOOOOOOOOOOOO
O             O
"""

# 1
four_rooms = """
OOOOO OOOOO
OOOOO OOOOO
OOOOOOOOOOO
OOOOO OOOOO
OOOOO OOOOO
 O    OOOOO
OOOOO   O  
OOOOO OOOOO
OOOOO OOOOO
OOOOOOOOOOO
OOOOO OOOOO
"""

# 2
two_rooms = """
OOOOOOOOOOOOOOOO
OOOOOOOOOOOOOOOO
OOOOOOOOOOOOOOOO
OOOOOOOOOOOOOOOO
OOOOOOOOOOOOOOOO
OOOOOOOOOOOOOOOO
OOOOOOOOOOOOOOOO
        OO      
OOOOOOOOOOOOOOOO
OOOOOOOOOOOOOOOO
OOOOOOOOOOOOOOOO
OOOOOOOOOOOOOOOO
OOOOOOOOOOOOOOOO
OOOOOOOOOOOOOOOO
OOOOOOOOOOOOOOOO
"""

# 2
hard_maze = """
OOOOOOOOOOOOOO
OOOOOOOOOOOOOO
      OO
OOOOO OOOOOOOO
OOOOO OOOOOOOO
OO     OO
OOOOOOOOOOOOOO
OOOOOOOOOOOOOO
OO OO OO
OO OO OO OOOOO
OO OO OO OOOOO
OO    OO    OO
OOOOO OOOOOOOO
OOOOO OOOOOOOO
"""

# 3
three_rooms = """
OOOOO OOOOO
OOOOO OOOOO
OOOOOOOOOOO
OOOOO OOOOO
OOOOO OOOOO
OOOOO 
OOOOO OOOOO
OOOOO OOOOO
OOOOOOOOOOO
OOOOO OOOOO
OOOOO OOOOO
"""

class Wrapper():
    pass

class GridWorld():
    """
    Simple GridWorld Class with similar attributes and methods as the discrete.DiscreteEnv inherited classes from openai/gym
    
    Parameters
    ==========
        id              : str
                          String id of gridworld. Expect one of {one_room, i_maze, four_rooms, two_rooms, hard_maze, three_rooms}.
                          
        terminal_states : dict [OPTIONAL, default = {}]
                          Dictionary with the form {state : reward}.
                          
        cost            : float, int [OPTIONAL, default = 0]
                          The movement cost.
    """
    metadata = {'render.modes': ['human', 'array']}
    
    def __init__(self, id : str, terminal_states : dict = {}, initial_states : list = [], cost : Union[float,int] = 0, max_steps : int = 500, p : float = 1.):
        ids = dict(one_room = one_room, i_maze = i_maze, four_rooms = four_rooms, two_rooms = two_rooms, hard_maze = hard_maze, three_rooms= three_rooms)
        if id not in ids:
            raise Exception(f'Expected one of {"{" + ", ".join(list(ids)) + "}"} but received "{id}"')
        
        self.spec       = Wrapper()
        self.spec.id    = id
        
        self._terminal_states = {}
        self._initial_states  = []
        
        self._grid      = self._to_grid(ids.get(id))
        self._reward    = np.ones_like(self._grid) * -cost
        self._max_steps = max_steps

        self._decode    = {i : np.array(state) for i, state in enumerate(zip(*np.where(self._grid)))}
        self._encode    = {tuple(v) : k for k, v in self._decode.items()}

        self.ACTIONS    = {0 : (1, 0), 1 : (0, 1), 2 : (-1, 0), 3 : (0, -1)}
        self.nA         = len(self.ACTIONS)
        self.nS         = len(self._decode)

        for terminal_state, reward in terminal_states.items():
            if isinstance(terminal_state, int):
                terminal_state = self(terminal_state)
            self._reward[terminal_state]          += reward
            self._terminal_states[terminal_state]  = reward
        
        for initial_state in initial_states:
            if isinstance(initial_state, int):
                initial_state = self(initial_state)
            self._initial_states.append(initial_state)
            
        self.P = {}
        for i, state in self._decode.items():
            self.P[i] = {}
            terminal  = tuple(state) in self._terminal_states
            for j, action in self.ACTIONS.items():
                if terminal:
                    self.P[i][j] = [(1.0, i, 0.0, True)]
                else:
                    self.P[i][j] = []
                    for k, action in self.ACTIONS.items():
                        intended  = j == k
                        new_state = state + action
                        prob      = p * intended + (1 - p) * (1 - intended)
                        self.P[i][j].append((prob, new_state, self._reward[self(new_state)], self(new_state) in self._terminal_states))
        
        self._reset = False
        
    def __call__(self, state):
        if isinstance(state, np.ndarray):
            state = tuple(state)
        if isinstance(state, list):
            return list(map(self, state))
        if isinstance(state, dict):
            return {self(key) : value for key, value in state.items()}
        if state in self._encode:
            return self._encode[state]
        return self._decode[state]

    def seed(self, *args, **kwargs):
        pass
    
    @staticmethod
    def _to_grid(string : str):
        split  = string.split('\n')[1:-1]                              # ignore first and last characters to allow '\n' at start and end
        n      = len(split)                                            # number of rows
        m      = max(map(len, split))                                  # number of columns
        grid   = np.zeros((n, m))
        for i, row in enumerate(split):
            grid[i, np.where(np.array([*row]) != ' ')[0]] = 1.
        return grid
    
    def reset(self):
        self._reset   = True
        if self._initial_states:
            self._state = self._initial_states[np.random.choice(len(self._initial_states))]
        else:
            self._state = np.random.randint(self.nS)
        self._pos     = self._states[self._state]
        self.terminal = False
        self._count   = 0
        return self._state
    
    def step(self, action : int):
        assert self._reset, "Cannot call env.step() before calling reset()"
        self._state   = self.P[self._state][action][0][1]
        self._pos     = tuple(self._states.get(self._state))
        reward        = self._reward[self._pos]
        self._count  += 1
        self.terminal = self._pos in self._terminal_states or self._count == self._max_steps
        return self._state, reward, self.terminal, {'prob' : 1.0}
    
    @property
    def desc(self):
        """
        String legend:
            "#" : wall
            " " : valid state
            "+" : valid state with positive terminal reward
            "-" : valid state with negative terminal reward
            
        Movement cost is ommited.
        """
        assert self._reset, "Cannot call env.desc() before calling reset()"
        grid            = np.empty_like(self._grid, dtype = str)
        grid[:]         = '#'
        grid[np.where(self._grid)] = ' '
        for terminal_state, reward in self._terminal_states.items():
            if reward < 0:
                grid[terminal_state] = '-'
            else:
                grid[terminal_state] = '+'
        grid[tuple(self._pos)] = 'A'
        return grid.astype('bytes') # to be consistent with other openai envs
    
    def render(self, mode = 'human', topology = True, ax = None):
        assert mode in self.metadata.get('render.modes')
        _grid   = np.pad(self.desc.astype(str), 1, mode = 'constant')
        grid    = np.zeros_like(_grid, dtype = float)
        S       = [' ', '+', '-', 'A']
        N       = range(1, 5)
        for s, n in zip(S, N):
            grid[np.where(_grid == s)] = n if topology else 1
            
        if mode == 'human':
            if ax is None: fig, ax = plt.subplots(figsize = (10, 10))
            ax.imshow(grid, cmap = 'magma')
            ax.axis('off')
            
            if topology:
                for s in S:
                    locs = zip(*np.where(_grid == s))
                    for loc in locs:
                        ax.annotate(s, loc[::-1], va = 'center', ha = 'center', fontsize = 20, color = 'k')
        else: # mode = 'array'
            return grid
