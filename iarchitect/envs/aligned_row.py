import numpy as np
from tf_agents.environments import py_environment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts

from .base_env import BaseEnv


class AlignedRowEnv(BaseEnv):
    def __init__(self,dimension,
                 action_float=False,
                 fail_on_same=True,
                 max_iter=50,
                 rewards = {
                     "already_filled":-10,
                     "max_iter":-10,
                     "new_value":1,
                     "success":10
                 }):
        super().__init__()
        self.dimension = dimension
        self._state = np.zeros((self.dimension,),dtype=np.int32)
        self._iter = 0
        self._max_iter = max_iter
        if action_float:
            self._action_spec = array_spec.BoundedArraySpec(
                shape=(), dtype=np.float32, minimum=-0.49, maximum=self.dimension+0.49, name='action')
        else:
            self._action_spec = array_spec.BoundedArraySpec(
                shape=(), dtype=np.int32, minimum=0, maximum=self.dimension-1, name='action')
        self._observation_spec = array_spec.BoundedArraySpec(
            shape=(self._state.shape[0],), dtype=np.int32, minimum=0, name='observation')

        self._episode_ended = False
        self.fail_on_same = fail_on_same
        self.rewards=rewards

    def _reset(self):
        self._state = np.zeros((self.dimension,),dtype=np.int32)
        self._iter = 0
        self._episode_ended = False
        return ts.restart(self.to_observation())


    def to_observation(self):
        return self._state.copy()

    def _step(self, action):
        """
        Remplit la action_ième case
        Termine si action déjà remplie
        """
        if self._episode_ended:
            # The last action ended the episode. Ignore the current action and start
            # a new episode.
            return self.reset()

        self._iter += 1

        # Make sure episodes don't go on forever.
        action_ = action.round().astype(int)
        assert action_ in list(range(self._state.shape[0]))

        if self._state[action_]==1:
            # DEJA RENSEIGNE
            reward = self.rewards["already_filled"]
            if self.fail_on_same:
                self._episode_ended = True
        else:
            self._state[action_]=1
            reward = self.rewards["new_value"]
            if self._state.sum()==self._state.shape[0]:
                reward = self.rewards["success"]
                self._episode_ended = True

        if not self.fail_on_same and self._iter>self._max_iter:
            reward = self.rewards["max_iter"]
            self._episode_ended = True

        if not self._episode_ended:
            result = ts.transition(
                self.to_observation(), reward=reward, discount=1)

        else:
            result = ts.termination(self.to_observation(), reward)
        return result

    def render(self,mode="human"):
        grill = self._state.reshape((int(self.dimension**0.5),int(self.dimension**0.5)))
        img = np.full((grill.shape[0]*16,grill.shape[1]*256),255)
        for r,c in itertools.product(range(grill.shape[0]),range(grill.shape[1])):
            img[r*16:r*16+16,c*16:c*16+16] = 255-grill[r,c]*255

        # print(img)
        # img = img * 255
        # print(img)
        # print(img.astype("uint8"),"in")
        # img = resize(img, (img.shape[0]*16,img.shape[1]*16)).astype("uint8")
        # print(img,"res")

        return img.astype('uint8')
