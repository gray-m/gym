""" Contains the implementation of ExploreGraphEnv """
#pylint: disable=W0613

from random import sample
from gym import Env
from gym.spaces import Discrete, Tuple
import numpy as np


def _generate_opponents(n_opponents, shape):
    raw = sample(range(np.prod(shape)), n_opponents)
    raw_to_point = lambda idx: _coord_from_index(idx, shape)
    return list(map(raw_to_point, raw))


def _coord_from_index(index, shape):
   return index//shape[1], index%shape[1]


def _left(loc, grid_shape):
    #print('LEFT!')
    if loc[1] == 0: # on the far left of the grid
        return loc

    return (loc[0], loc[1] - 1)


def _right(loc, grid_shape):
    #print('RIGHT!')
    if loc[1] == grid_shape[1] - 1:
        return loc

    return (loc[0], loc[1] + 1)


def _up(loc, grid_shape):
    # print('UP!')
    if loc[0] == 0: # in row 0, so no top motion possible
        return loc

    return (loc[0] - 1, loc[1])


def _down(loc, grid_shape):
    #print('DOWN!')
    if loc[0] == grid_shape[0] - 1: # in bottom row, so no bottom
        return loc

    return (loc[0] + 1, loc[1])


MOVES = [_left, _right, _up, _down, lambda loc, _: loc]
N_MOVES = 5


def _move_opponents(opponent_locs, grid_shape):
    """
    Move opponents during a step call based on the opponent locations and the
    shape of the grid.

    Args:
        grid_shape: a tuple of dimensions of a grid
        opponent_locs: a scalar np.array of locations in a grid of shape grid_shape
    """
    move_directions = [np.random.randint(N_MOVES) for _ in range(len(opponent_locs))]
    moves_to_make = map(lambda n: MOVES[n], move_directions)
    pairs = zip(moves_to_make, opponent_locs)
    return list(map(lambda pair: pair[0](pair[1], grid_shape), pairs))


class ExploreGraphEnv(Env):
    """
    An environment that contains opponents which perform a random walk
    on a graph, in which the agent's goal is to survive as long as
    possible ('survival' is not colliding with an opponent).

    Observations consist of a list of locations. The head of the list
    is the agent's location, and the tail is the list of opponent
    locations.
    """

    def __init__(self, n_opponents=20, shape=(10, 10)):
        """
        Initialize the environment given a number of opponents
        and a shape for the grid (the graph will be modeled as
        a grid for now)

        Args:
            n_opponents: The number of opponents to place randomly
                         on the graph
            shape: the dimensions of the grid
        """

        self._n_opponents = n_opponents
        self._agent_loc = None
        self._opponent_locs = None
        self._shape = shape

        self.action_space = Discrete(N_MOVES)
        self.observation_space = Tuple((Discrete(np.prod(shape)),) * (n_opponents + 1))
        self.observation_space.shape = (n_opponents + 1,)

        self.spec = None

        self._done = False


    def _get_observation(self):
        return (self._agent_loc,) + tuple(self._opponent_locs)


    def _reset(self):
        self._opponent_locs = _generate_opponents(self._n_opponents, self._shape)
        # make sure the agent doesn't start off dead
        self._agent_loc = _coord_from_index(np.random.randint(np.prod(self._shape)), self._shape)
        while self._agent_loc in self._opponent_locs:
            self._agent_loc = _coord_from_index(np.random.randint(np.prod(self._shape)), self._shape)
        return [self._agent_loc] + self._opponent_locs


    def _step(self, action):
        self._opponent_locs = _move_opponents(self._opponent_locs, self._shape)
        self._agent_loc = MOVES[action](self._agent_loc, self._shape)
        
        if not self._done:
            self._done = self._agent_loc in self._opponent_locs
        
        reward = 1.0 if not self._done else 0.0
        return self._get_observation(), reward, self._done, {}


