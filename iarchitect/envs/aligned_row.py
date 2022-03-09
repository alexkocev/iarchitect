import itertools

import numpy as np
from matplotlib import pyplot as plt
from tf_agents.environments import py_environment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts
from skimage.transform import resize

from .base_env import BaseEnv
from ..render import npemojis, image_from_text


class AlignedRowEnv(BaseEnv):
    def __init__(self,dimension,
                 observation_1D=True,
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
        self.emojis = npemojis(1,with_empy=True)
        self._state = np.zeros((self.dimension,),dtype=np.int32)
        self._iter = 0
        self._max_iter = max_iter

        if action_float:
            self._action_spec = array_spec.BoundedArraySpec(
                shape=(), dtype=np.float32, minimum=-0.49, maximum=self.dimension+0.49, name='action')
        else:
            self._action_spec = array_spec.BoundedArraySpec(
                shape=(), dtype=np.int32, minimum=0, maximum=self.dimension-1, name='action')

        self.observation_1D = observation_1D
        if not self.observation_1D:
            self._observation_spec = array_spec.BoundedArraySpec(
                shape=(int(self.dimension**0.5),int(self.dimension**0.5)), dtype=np.int32, minimum=0, name='observation')
        else:
            self._observation_spec = array_spec.BoundedArraySpec(
                shape=(self._state.shape[0],), dtype=np.int32, minimum=0, name='observation')

        self._episode_ended = False
        self.fail_on_same = fail_on_same
        self.rewards=rewards
        self._last_action = None
        self._last_reward = 0


    def _reset(self):
        self._state = np.zeros((self.dimension,),dtype=np.int32)
        self._iter = 0
        self._episode_ended = False
        self._last_action = None
        self._last_reward = 0
        self._last_position = None
        return ts.restart(self.to_observation())


    def to_observation(self):
        if not self.observation_1D:
            return self._state.reshape((int(self.dimension**0.5),int(self.dimension**0.5))).copy()
        else:
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

        self._last_action = 1
        self._last_position = action_
        self._last_reward = reward
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


    def render_strings(self):
        r = int(self.dimension**0.5)
        c = r
        assert r*c == self.dimension

        grid = [" ".join(self.emojis[self._state.reshape((r,c))[i,:]].flat) for i in range(r)]
        la = self.emojis[self._last_action] if self._last_action is not None else "❌"
        at = f"@ {','.join(map(str,np.unravel_index(self._last_position, (r, c))))}" \
            if self._last_position is not None else "@ 0,0"

        last = [f"Last : {la} {at}",f"\t-> R : {self._last_reward}"]
        return grid,last

    def render_image(self):
        grid,last = self.render_strings()
        texts = grid+last
        fnt_sizes = [40]*len(texts)
        n = (len(last))
        fnt_sizes[-n:] = [int(40/3)]*n
        im = image_from_text(texts,fnt_sizes,max_size=546)
        return im

    def render(self,mode="human"):
        if mode=="human":
            grid,last = self.render_strings()
            return "\n".join(grid+last)
        else:
            return np.array(self.render_image())
