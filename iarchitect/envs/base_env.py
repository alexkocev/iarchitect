import numpy as np
from tf_agents.environments import py_environment

class BaseEnv(py_environment.PyEnvironment):

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def to_observation(self):
        return self._state.copy()