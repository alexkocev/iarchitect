import itertools

import numpy as np
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts

from iarchitect.envs.base_env import BaseEnv
from iarchitect.render import npemojis, image_from_text



def make_nemesis(n,diag_only=False):
    a = np.zeros((n,n))
    if not diag_only:
        for i in range(a.shape[0]):
            a[i,:] = np.random.normal(0,0.5,(a.shape[1]))
        a[a<-1]=-1
        a[a>1]=1
    for i in range(a.shape[0]):
        a[i,i] = 1
    a[np.abs(a)<0.3]=0
    a[np.abs(a)<0.3]=0
    for r in range(a.shape[0]):
        for c in range(r,a.shape[1]):
            a[r,c] = a[c,r]
    return a

class WindowEnv(BaseEnv):
    def __init__(self,dimension,tuiles,
                 nemesis=None,
                 random_reset=True,
                 max_species_reset=5,
                 render_dims=None,
                 action_float=False,
                 action_shape_one=False,
                 strategie=1,
                 discount=1,
                 random_next_position=False,
                 only_diag_nemesis=True):
        super().__init__()
        self.random_next_position = random_next_position
        self.nemesis = nemesis
        self.nemesis_resetable = True
        self.only_diag_nemesis = only_diag_nemesis
        if nemesis is not None:
            assert strategie>100
            assert nemesis.shape[0]==len(tuiles)                # <---- ak UPDATE
            assert nemesis.shape[0]==nemesis.shape[1]
            self.nemesis_resetable = False
            self.nemesis = nemesis +1
        self.strategie = strategie
        self.discount = discount

        self.dimension = dimension
        self.render_dims = render_dims

        self.quotas = None
        self.quotas_resetable = True
        if render_dims is not None:
            assert render_dims[0]*render_dims[1]==dimension
            self.render_dims = tuple(map(int,self.render_dims))

        self.tuiles = tuiles

        self.random_reset = random_reset
        self.max_species_reset = max_species_reset
        self.emojis = npemojis(tuiles.shape[0],with_empy=True)

        # self.quotas[self.quotas==0]=1e-6

        self._state = np.zeros((self.dimension,),dtype=np.int)

        self.action_float =action_float
        self.action_shape_one = action_shape_one
        if action_float:
            if action_shape_one:
                self._action_spec = array_spec.BoundedArraySpec(
                    shape=(1,), dtype=np.float32, minimum=0, maximum=len(self.tuiles)-0.5001, name='action')
            else:
                self._action_spec = array_spec.BoundedArraySpec(
                shape=(), dtype=np.float32, minimum=0, maximum=len(self.tuiles)-0.5001, name='action')
        else:
        # /!\ => action = especes
            if action_shape_one:
                self._action_spec = array_spec.BoundedArraySpec(
                shape=(1,), dtype=np.int, minimum=0, maximum=len(self.tuiles)-1, name='action')
            else:
                self._action_spec = array_spec.BoundedArraySpec(
                    shape=(), dtype=np.int, minimum=0, maximum=len(self.tuiles)-1, name='action')

        # ETAPE 1 : OBSERVATION = REMPLISSAGE (ON RAJOUTE UNE CASE POUR LES CASES VIDES)

        if self.strategie==5:
            # Normalisation des observations
            self._observation_spec = array_spec.BoundedArraySpec(
                shape=(self.tuiles.shape[0]+1,), dtype=np.float32, minimum=-1,maximum=1, name='observation')
        elif self.strategie == 101:
            # On retourne un tableau a deux dimensiosn
            self._observation_spec = array_spec.BoundedArraySpec(
                shape=(self.tuiles.shape[0]+1,self.tuiles.shape[0]), dtype=np.float32, minimum=-1,maximum=1, name='observation')
        elif self.strategie==102:
            # On retourne un tableau a deux dimensiosn
            self._observation_spec = array_spec.BoundedArraySpec(
                shape=(self.tuiles.shape[0]+2,self.tuiles.shape[0]), dtype=np.float32, minimum=-1,maximum=1, name='observation')
        else:
            self._observation_spec = array_spec.BoundedArraySpec(
                shape=(self.tuiles.shape[0]+1,), dtype=np.float32, minimum=0, name='observation')

        self._episode_ended = False
        self._next_position = None
        self._next_neighbours = None
        self._last_value = 0


    def set_nemesis(self,nemesis,resetable=False):
        assert nemesis.shape[0]==self.tuiles.shape[0],f"{self.tuiles.shape[0]} attendu, {nemesis.shape[0]} vu"
        self.nemesis = nemesis
        self.nemesis_resetable = resetable
        self.next_position()


    def set_quotas(self,quotas,resetable=False):
        assert quotas.shape[0]==self.tuiles.shape[0],f"{self.tuiles.shape[0]} attendu, {quotas.shape[0]} vu"
        self.quotas = quotas
        self.quotas_demandes_mask = self.quotas!=0.0
        self.quotas_resetable = resetable
        self.taux_mask_quotas = np.insert(self.quotas_demandes_mask,0,False) # Taux a une case de plus que quota. On ajout False ?? la fin
        self._taux = self.taux_remplissage()
        self.next_position()


    def next_position(self):
        """
        Determine la prochaine position vide
        :return: None si la grille est full
        """
        if self.random_next_position:
            pos = None
            indices = np.argwhere(self._state==0)
            np.random.shuffle(indices)
            if indices.shape[0]:
                pos = np.unravel_index(indices[0],self._state.shape)
        else:
            pos = np.unravel_index(np.argmin(self._state),self._state.shape)
            if self._state[pos]!=0:
                pos = None
        self._next_position = pos
        self.next_neighbours()

    def _reset(self):
        """
        Retourne une nouvelle grille avec au moins une position non vide
        :return:
        """
        if self.random_reset or self.max_species_reset is not None:
            if self.max_species_reset is None:
                self._state = np.random.randint(1,len(self.tuiles)+1,(self.dimension,),np.int)
                nb_zeros = np.random.randint(0,self.dimension)
                zero_indices = np.random.randint(0,self.dimension,(nb_zeros,))
                self._state[zero_indices] = 0
            else:
                self._state = np.zeros((self.dimension,),dtype=np.int)
                species_indexes = np.random.randint(0,self.dimension,(self.max_species_reset,))
                self._state[species_indexes] = np.random.randint(1,len(self.tuiles)+1,(self.max_species_reset,))
        else:
            self._state = np.zeros((self.dimension,),dtype=np.int)

        self.next_position()
        if self._next_position is None:
            # LA GRILLE EST DEJA PLEINE
            return self._reset()

        if self.quotas_resetable:
            # LOI NORMALE POUR DEFINIR LES QUOTAS
            self.set_quotas(np.abs(
                (np.random.normal(0,
                                  0.07,
                                  size=(len(self.tuiles),))*self.dimension).astype(int)))

        if self.nemesis_resetable:
            self.set_nemesis(
                make_nemesis(len(self.tuiles),diag_only=self.only_diag_nemesis)
            )

        self._taux = self.taux_remplissage()
        self._episode_ended = False
        self._last_value = self.evaluate_grid()
        self._last_position = None
        self._last_reward = 0
        self._last_action = None
        return ts.restart(self.to_observation())


    def taux_remplissage(self):
        """
        :return: [] + taux_pour les esp??ces
        """
        taux = np.full((len(self.tuiles)+1,),0.0,dtype=np.float32) # Taux a une case de plus que quota ou tuiles
        ind,c = np.unique(self._state,return_counts=True)
        taux[ind] = c  # Ind sont bien compris entre 1 et n_tuiles, car lues dans state
         # Pour les quotas impos??s, ie diff??rent de zeros
        taux[self.taux_mask_quotas] = taux[self.taux_mask_quotas]/self.quotas[self.quotas_demandes_mask]

        if self.strategie==5:
            # Normalisation des observations
            taux[~(self.taux_mask_quotas)] = taux[~(self.taux_mask_quotas)]/0.1
            taux[~(self.taux_mask_quotas) & (taux==0)] = 1+1e-6
            taux[taux==0] = 1e-6
            mask = taux<1
            taux[mask] = np.tanh(-(np.power(taux[mask],-1)-1))
            taux[~(mask)] = np.tanh(taux[~(mask)]-1)
            # TODO QUE FAIRE DE TAUX[0]
        else:
            taux[taux>1.0] = 1.0
            taux[~(self.taux_mask_quotas)] = 1.0

        # TODO MULTIPLIER PAR LES RENDEMENTS
        # TODO CONFIRMER QUE QUOTOS
        return taux

    def evaluate_grid(self):
        compteurs = np.full((len(self.tuiles),),0.0,dtype=np.float32)
        ind,c = np.unique(self._state,return_counts=True) # SHIFT DE -1 pour retrouver des indices qui commencent par 0 et supprimer les cases vides
        c = c[ind!=0]
        ind = ind[ind!=0]-1
        compteurs[ind]=c

        value = 0
        # quotas demand??s
        mask_quota_nok = \
            compteurs[self.quotas_demandes_mask]<self.quotas[self.quotas_demandes_mask]
        mask_quota_ok = \
            compteurs[self.quotas_demandes_mask]>=self.quotas[self.quotas_demandes_mask]
        if mask_quota_nok.any():
            # DES QUOTAS SONT ENCORE VIDES => ON NE COMPTE QUE LES QUOTAS OK + LES QUOTAS NOK EN COURS
            value += self.quotas[self.quotas_demandes_mask][mask_quota_ok].sum()
            value += compteurs[self.quotas_demandes_mask][mask_quota_nok].sum()
        else:
            # TOUS LES QUOTAS REMPLIS ON COMPTE TOUT
            value += compteurs.sum()
            assert value == (self._state!=0).sum()
        # TODO AJOUTER LES RENDEMENTS
        return value


    def to_observation(self):
        if self.strategie==3:
            ret = self._taux.copy()
            # TODO Essayer
            # # On retourne au programme dans le cas de la strat??gie 3 le nombre de case restante
            # ret[0] = (self._state == 0).sum()
            return ret
        elif self.strategie==101:
            ret = np.vstack((self.nemesis,self._next_neighbours)).astype(np.float32)
            return ret
        elif self.strategie==102:
            ret = np.vstack((self.nemesis,self._next_neighbours,self._taux[1:])).astype(np.float32)
            return ret
        else:
            return self._taux.copy()

    def next_neighbours(self):         # <-------------------------ak UPDATE
        # Convert coordinate of _next_position from 1D to 2D grid
        self._next_neighbours = np.full((len(self.tuiles),),0,dtype=np.float32)
        if self._next_position is not None:
            xy = self._next_position[-1]
            n = int(self.dimension**0.5)
            x = xy // n
            y = xy % n

            # Reshape the _state as a 2D grid to determine closest neighbours
            state_reshaped = self._state.reshape((n, -1))

            neighbours = [] # Tableau des voisins (commence par 1 pour le premier element)

            # north
            if y-1 >0:
                neighbours.append(state_reshaped[y-1, x])

            # north_east
            if x+1 <n and y-1 >0:
                neighbours.append(state_reshaped[y-1, x+1])

            # east
            if x+1 <n:
                neighbours.append(state_reshaped[y, x+1])

            # south_east
            if x+1 <n and y+1 <n:
                neighbours.append(state_reshaped[y+1, x+1])

            # south
            if y+1 <n:
                neighbours.append(state_reshaped[y+1, x])

            # south_west
            if x-1 >0 and y+1 <n:
                neighbours.append(state_reshaped[y+1, x-1])

            # west
            if x-1 >0:
                neighbours.append(state_reshaped[y, x-1])

            # north_west
            if x-1 >0 and y-1 >0:
                neighbours.append(state_reshaped[y-1, x-1])

            neighbours  = np.array(neighbours)-1 # SHIFT POUR AVOIR DES LEGUMES QUI COMMENCENT A 0
            ind,count_ = np.unique(neighbours[neighbours!=-1],return_counts=True)
            self._next_neighbours[ind] = count_
        self._next_neighbours = self._next_neighbours/9*2-1 # Normalisation car 9 voisins
            ################################################


    def _step(self, action):
        """
        Remplit la action_i??me case
        Termine si action d??j?? remplie
        """
        action = int(np.round(action).astype(int))
        if self._episode_ended:
            # The last action ended the episode. Ignore the current action and start
            # a new episode.
            return self.reset()

        # ACTION ENTRE 0 et N_Tuile-1
        # DANS LE STATE LES ESPECES SONT REPERERES PAR i_TUILE + 1 car 0 Pour case vides
        espece_vue = action + 1

        self._episode_ended = False
        reward = 0

        # print(self._state,self._next_position)
        self._state[self._next_position] = espece_vue # POSE DU LEGUME
        # print(self._state)
        new_taux = self.taux_remplissage() # NOUVEAU TAUX, LES ANCIENS SONT ENCORE DANS self._taux
        new_value = self.evaluate_grid()
        self._last_position = self._next_position
        self._last_neighbours = self._next_neighbours.copy()
        self.next_position()


        # STRATEGIE = DELTA_VALEUR
        if self.strategie == 1 :
            reward = new_value-self._last_value
            self._episode_ended = self._next_position is None
            # if self._next_position is None:
            #     reward = 0.5
            #     self._episode_ended = True



        elif self.strategie == 2:
            # STRATEGIE = VALEUR SELON TAUX DE REMPLISSAGE
            if self._taux[espece_vue]<1.0 and self.quotas[action]>0.0:
                # ON AUGMENTE UN QUOTA A REMPLIR
                reward = 0.1
                if new_taux[espece_vue]>1.0:
                    # ON LE REMPLI COMPLETEMENT
                    reward = 0.2
            else:
                reward = -0.1 # INCITATION A REMPLIR DES TAUX PROCHES DE PLEINS

            if self._next_position is None:
                reward = 0.5
                self._episode_ended = True
        elif self.strategie == 3:
            reward = -0.1
            if self._next_position is None:
                reward = self.evaluate_grid()
                self._episode_ended = True
        elif self.strategie == 5:
            ### Normalisation des observations
            reward = 0.1
            oldtx = self._taux[espece_vue]
            if oldtx<0:
                reward = reward
            else:
                reward = -reward-oldtx

            if self._next_position is None:
                reward = 1
                self._episode_ended = True
        elif self.strategie in [101,102]:
            # gestion des nemesis
            rewardnemesis = (self.nemesis[action,:]*((self._last_neighbours+1)/2)*1).sum()
            reward =rewardnemesis
            if self.strategie==102:
                # gestion des taux
                reward = 0.1
                oldtx = self._taux[espece_vue]
                if oldtx<0:
                    reward = reward
                else:
                    reward = -reward-oldtx
                if rewardnemesis*reward>=0:
                    reward = rewardnemesis+reward
                else:
                    reward = -(abs(rewardnemesis) + abs(reward))


            if self._next_position is None:
                if self.strategie==101:
                    reward = 1
                self._episode_ended = True

        else:
            raise Exception


        self._taux = new_taux # ON ENREGISTRE LES NOUVEAUX TAUX
        self._last_value = new_value
        # print(reward,self._episode_ended,action,self._next_position)

        self._last_action = action
        self._last_reward = reward
        if not self._episode_ended:
            result = ts.transition(
                self.to_observation(), reward=reward, discount=self.discount)
        else:
            result = ts.termination(self.to_observation(), reward)
        return result




    def render_strings(self):
        compteurs = np.full((len(self.tuiles),),0.0,dtype=np.float32)
        ind,c = np.unique(self._state,return_counts=True) # SHIFT DE -1 pour retrouver des indices qui commencent par 0 et supprimer les cases vides
        c = c[ind!=0]
        ind = ind[ind!=0]-1
        compteurs[ind]=c
        taux = compteurs
        taux[self.quotas_demandes_mask] = taux[self.quotas_demandes_mask] / self.quotas[self.quotas_demandes_mask]
        taux[~(self.quotas_demandes_mask)] = taux[~(self.quotas_demandes_mask)] / 0.1

        r,c = self.render_dims
        grid = [" ".join(self.emojis[self._state.reshape((r,c))[i,:]].flat) for i in range(r)]
        q = [f"{self.emojis[i+1]} : {q}" for i,q in enumerate(self.quotas)]
        tx = [f"{self.emojis[i+1]} : {t*100:3.0f}%" for i,t in enumerate(taux)]
        la = self.emojis[self._last_action+1] if self._last_action is not None else ""
        at = f"@ {','.join(map(str,np.unravel_index(self._last_position, (r, c))))}" if self._last_position is not None else ""

        last = [f"Last : {la} {at} -> R : {self._last_reward:4.3f}"]

        # Groupes quotas par ligne
        pos = 0
        quotas = ["Quotas : "]
        while pos<len(q):
            quotas.append(" ".join(q[pos:min(pos+r+1,len(q))]))
            pos = pos+r+1
        # Groupes quotas par ligne
        pos = 0
        taux = ["Remplissage : "]
        while pos<len(tx):
            taux.append(" ".join(tx[pos:min(pos+r+1,len(tx))]))
            pos = pos+r+1

        total = f"Rendement : {self.evaluate_grid():4.2f} ??? "
        return grid,last,quotas ,taux , [total]

    def render_image(self):
        grid,last,quotas ,taux , totaux = self.render_strings()
        texts = grid+last+quotas+taux+totaux
        fnt_sizes = [40]*len(texts)
        n = (len(last)+len(quotas)+len(taux)+len(totaux))
        fnt_sizes[-n:] = [int(40/3)]*n
        im = image_from_text(texts,fnt_sizes,max_size=546)
        return im

    def render(self,mode="human"):
        if mode=="human":
            grid,last,quotas ,taux ,totaux = self.render_strings()
            return "\n".join(grid+last+quotas +taux +totaux)
        else:
            return np.array(self.render_image())
