from operator import attrgetter, itemgetter

import matplotlib.pyplot as plt
import numpy as np

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