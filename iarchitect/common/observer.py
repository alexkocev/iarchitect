import pickle
from operator import attrgetter, itemgetter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tensorflow import Tensor

class ObserverTrajectory:
    def __init__(self,verbose=False):
        self.results = []
        self.verbose = verbose

    def __call__(self,traj):
        self.results.append(traj)
        if self.verbose:
            print("\tMyObserver:",traj.observation,traj.reward)

    def trajectories(self):
        return self.results

    def commonattr(self,attr):
        return list(map(attrgetter(attr),self.trajectories()))

    def rewards(self):
        return self.commonattr("reward")

    def observations(self):
        return self.commonattr("observation")

    def plot_reward(self, ax=None, slice_=slice(None,None,None)):
        n = len(self.results)
        x = list(range(n))
        if ax is None:
            fig,ax = plt.subplots()
        results = self.results[slice_]
        ax.plot([t.reward.numpy() for t in results])
        return ax

    def plot_action(self, ax=None, slice_=slice(None,None,None)):
        n = len(self.results)
        x = list(range(n))
        if ax is None:
            fig,ax = plt.subplots()
        results = self.results[slice_]

        toplot = np.array([t.action.numpy() for t in results])
        for c in range(toplot.shape[1]):
            ax.plot(toplot[:,c])
        return ax

    def save(self,folder,suff):
        p = Path(folder)
        p.mkdir(exist_ok=True)
        p = p / ("observer_"+suff)
        with open(p,"wb") as f:
            pickle.dump(self.results,f)

    @classmethod
    def load(cls,folder,suff):
        obs = cls()
        p = Path(folder)
        p = p / ("observer_"+suff)
        with open(p,"rb") as f:
            obs.results = pickle.load(f)
        return obs

