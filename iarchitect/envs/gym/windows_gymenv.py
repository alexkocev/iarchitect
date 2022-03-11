import numpy as np
import itertools
import stable_baselines3

# Environement
import gym
from gym import spaces

# Evaluate the environement
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.results_plotter import load_results, ts2xy, plot_results
from stable_baselines3.common.monitor import Monitor

# Agent
from stable_baselines3 import A2C,PPO



class WindowGymEnv(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(self,dimension,tuiles,quotas):
        super().__init__()
        assert quotas.shape[0]==tuiles.shape[0]
        self.dimension = dimension
        self.tuiles = tuiles
        self.quotas = quotas

        self._state = np.zeros((self.dimension,),dtype=np.int8)

        # /!\ => action = especes
        self.action_space = spaces.Discrete(self.tuiles.shape[0])

        # ETAPE 1 : OBSERVATION = REMPLISSAGE (ON RAJOUTE UNE CASE POUR LES CASES VIDES)
        self.observation_space = spaces.Box(low=0,
                                            high=np.inf,
                                            shape=(self.tuiles.shape[0]+1,),
                                            dtype=np.float32)

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


    def taux_remplissage(self):
        """
        Détermine le taux de remplissage des quotas
        :return: [] + taux_pour les espèces
        """
        taux = np.full((len(self.tuiles)+1,),0.0,dtype=np.float32) # Taux a une case de plus que quota ou tuiles
        ind,c = np.unique(self._state,return_counts=True)
        taux[ind] = c  # Ind sont bien compris entre 1 et n_tuiles, car lues dans state
        mask = self.quotas!=0.0 # Pour les quotas imposés, ie différent de zeros
        mask_tx= np.insert(mask,0,False) # Taux a une case de plus que quota. On ajout False à la fin
        taux[mask_tx] = taux[mask_tx]/self.quotas[mask]
        taux[~(mask_tx)] = 1.0

        return taux


    def step(self, action):
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

        info = {}
        espece_vue = action + 1

        self._episode_ended = False
        reward = 0

        self._state[self._next_position] = espece_vue # POSE DU LEGUME
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

        return self.to_observation(), reward, self._episode_ended, info


    def reset(self):
        self._state = np.random.randint(1,len(self.tuiles)+1,(self.dimension,),np.int8)

        nb_zeros = np.random.randint(0,self.dimension)
        zero_indices = np.random.randint(0,self.dimension,(nb_zeros,))
        self._state[zero_indices] = 0

        self._next_position = self.next_position()
        if self._next_position is None:
            return self.reset()

        self._taux = self.taux_remplissage()
        self._episode_ended = False

        return self.to_observation()


    def to_observation(self):
        return self._taux.copy()


    def render(self,mode="human"):
        grill = self._state.reshape((int(self.dimension**0.5),int(self.dimension**0.5)))
        img = np.full((grill.shape[0]*16,grill.shape[1]*256),255)
        for r,c in itertools.product(range(grill.shape[0]),range(grill.shape[1])):
            img[r*16:r*16+16,c*16:c*16+16] = 255-grill[r,c]*255

        return img.astype('uint8')
