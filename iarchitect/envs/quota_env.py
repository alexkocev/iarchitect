import imp
import numpy as np
import itertools
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts

from .base_env import BaseEnv


quota_laitue = 3
quota_carotte = 2
quota_broccoli = 0
quota_tomate = 1
quota_chou = 1
quota_haricot = 0
quota_patate = 4
quota_ail = 3
quota_oignon = 0
quota_courgette = 1


ensemble_des_tuiles = dict()

ensemble_des_tuiles[1] = ['tomate',4,3,quota_tomate]
ensemble_des_tuiles[2] = ['carotte',2,5,quota_carotte]
ensemble_des_tuiles[3] = ['broccoli',6,1,quota_broccoli]
ensemble_des_tuiles[4] = ['chou',3,3,quota_chou]
ensemble_des_tuiles[5] = ['laitue',2,1,quota_laitue]
ensemble_des_tuiles[6] = ['haricot',6,2,quota_haricot]
ensemble_des_tuiles[7] = ['patate',2,4,quota_patate]
ensemble_des_tuiles[8] = ['ail',8,1,quota_ail]
ensemble_des_tuiles[9] = ['oignon',8,1,quota_oignon]
ensemble_des_tuiles[10] = ['courgette',1,2,quota_courgette]

nb_ensemble_des_tuiles = len(ensemble_des_tuiles)


ensemble_des_rewards = []
ensemble_des_quotas = []

for k,v in ensemble_des_tuiles.items():
    ensemble_des_rewards.append(v[2])

for k,v in ensemble_des_tuiles.items():
    ensemble_des_quotas.append(v[-1])


nb_tuiles = nb_ensemble_des_tuiles
dim_x = 25
dim_y = 1
ensemble_des_possibles = dict()

compteur = -1
for x,y,t in itertools.product(range(dim_x),range(dim_y),range(1, nb_tuiles+1)):
    compteur += 1
    ensemble_des_possibles[compteur] = [x,y,t]

nb_ensemble_des_possibles = len(ensemble_des_possibles)
print(nb_ensemble_des_possibles)
ensemble_des_possibles


class QuotaEnv(BaseEnv):
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


    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def _reset(self):
        self._state = np.zeros((self.dimension,),dtype=np.int32)
        self._iter = 0
        self._episode_ended = False
        return ts.restart(self.to_observation())


    def to_observation(self):
        if not self.observation_1D:
            return self._state.reshape((int(self.dimension**0.5),int(self.dimension**0.5))).copy()
        else:
            self._state.copy()

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
