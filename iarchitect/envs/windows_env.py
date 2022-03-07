import itertools

import numpy as np
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts

from iarchitect.envs.base_env import BaseEnv


class WindowEnv(BaseEnv):
    def __init__(self,dimension,tuiles,quotas):
        super().__init__()
        assert quotas.shape[0]==tuiles.shape[0]
        self.dimension = dimension
        self.tuiles = tuiles
        self.quotas = quotas
        # self.quotas[self.quotas==0]=1e-6

        self._state = np.zeros((self.dimension,),dtype=np.int8)

        # /!\ => action = especes
        self._action_spec = array_spec.BoundedArraySpec(
                shape=(), dtype=np.int, minimum=0, maximum=len(self.tuiles)-1, name='action')

        # ETAPE 1 : OBSERVATION = REMPLISSAGE (ON RAJOUTE UNE CASE POUR LES CASES VIDES)
        self._observation_spec = array_spec.BoundedArraySpec(
                shape=(self.tuiles.shape[0]+1,), dtype=np.float32, minimum=0, name='observation')

        self._episode_ended = False
        self._next_position = self.next_position()
        self._taux = self.taux_remplissage()

    def next_position(self):
        """
        Determine la prochaine position vide
        :return: None si la grille est full
        """
        pos = np.unravel_index(np.argmin(self._state),self._state.shape)
        if self._state[pos]!=0:
            return None
        return pos

    def _reset(self):
        """
        Retourne une nouvelle grille avec au moins une position non vide
        :return:
        """
        self._state = np.random.randint(1,len(self.tuiles)+1,(self.dimension,),np.int8)

        nb_zeros = np.random.randint(0,self.dimension)
        zero_indices = np.random.randint(0,self.dimension,(nb_zeros,))
        self._state[zero_indices] = 0

        self._next_position = self.next_position()
        if self._next_position is None:
            return self._reset()

        self._taux = self.taux_remplissage()
        self._episode_ended = False
        return ts.restart(self.to_observation())


    def taux_remplissage(self):
        """

        :return: [] + taux_pour les espèces
        """
        taux = np.full((len(self.tuiles)+1,),0.0,dtype=np.float32) # Taux a une case de plus que quota ou tuiles
        ind,c = np.unique(self._state,return_counts=True)
        taux[ind] = c  # Ind sont bien compris entre 1 et n_tuiles, car lues dans state
        mask = self.quotas!=0.0 # Pour les quotas imposés, ie différent de zeros
        mask_tx= np.insert(mask,0,False) # Taux a une case de plus que quota. On ajout False à la fin
        taux[mask_tx] = taux[mask_tx]/self.quotas[mask]
        taux[~(mask_tx)] = 0.0
        # TODO MULTIPLIER PAR LES RENDEMENTS
        # TODO CONFIRMER QUE QUOTOS
        return taux


    def to_observation(self):
        return self._taux.copy()

    def _step(self, action):
        """
        Remplit la action_ième case
        Termine si action déjà remplie
        """
        if self._episode_ended:
            # The last action ended the episode. Ignore the current action and start
            # a new episode.
            return self.reset()

        # ACTION ENTRE 0 et N_Tuile-1
        # DANS LE STATE LES ESPECES SONT REPERERES PAR i_TUILE + 1 car 0 Pour case vides

        espece_vue = action + 1

        self._episode_ended = False
        reward = 0

        self._state[self._next_position] = action # POSE DU LEGUME
        new_taux = self.taux_remplissage() # NOUVEAU TAUX, LES ANCIENS SONT ENCORE DANS self._taux
        self._next_position = self.next_position()


        if self._taux[espece_vue]<1.0 and self.quotas[action]>0.0:
            # ON AUGMENTE UN QUOTA A REMPLIR
            reward = 0.1
            if new_taux[espece_vue]>1.0:
                # ON LE REMPLI COMPLETEMENT
                reward = 0.2
        else:
            reward = 0 # INCITATION A REMPLIR DES TAUX PROCHES DE PLEINS
        if self._next_position is None:
            reward = 0.5
            self._episode_ended = True

        self._taux = new_taux # ON ENREGISTRE LES NOUVEAUX TAUX
        if not self._episode_ended:
            result = ts.transition(
                self.to_observation(), reward=reward, discount=1)
        else:
            result = ts.termination(self.to_observation(), reward)
        return result

    def render(self):
        grill = self._state.reshape((int(self.dimension**0.5),int(self.dimension**0.5)))
        img = np.full((grill.shape[0]*16,grill.shape[1]*256),255)
        for r,c in itertools.product(range(grill.shape[0]),range(grill.shape[1])):
            img[r*16:r*16+16,c*16:c*16+16] = 255-grill[r,c]*255

        return img.astype('uint8')