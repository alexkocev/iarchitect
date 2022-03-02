import matplotlib.pyplot as plt

from tensorflow import Tensor

class ObserverTrajectory:
    def __init__(self,verbose=False):
        self.results = []
        self.verbose = verbose

    def __call__(self,traj):
        obs = traj.observation
        if isinstance(obs,Tensor):
            obs = obs.numpy().copy()
        else:
            obs = obs.copy()
        self.results.append({"observation":obs,"reward":traj.reward,"traj":traj})
        if self.verbose:
            print("\tMyObserver:",traj.observation,traj.reward)

    def plot_reward(self, ax=None, slice_=slice(None,None,None)):
        n = len(self.results)
        x = list(range(n))
        if ax is None:
            fig,ax = plt.subplots()
        results = self.results[slice_]
        ax.plot([t.get("reward") for t in results])
        return ax