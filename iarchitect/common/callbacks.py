import itertools

from IPython.core.display import clear_output as clo, display
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec


def output_updater(*args,clear_output=True):
    """
    Pour mise à jour des args
    :param args: args à mettre à jour dans l'output sous la cellule
    :param clear_output:
    :return:
    """
    def update(step,trainer):
        if clear_output:
            clo(wait = True)
        for a in args:
            display(a)
    return update


def update_plotter(fig,plot_obs=None,plot_traj=None,slice_=slice(-50,None,None)):
    """

    :param fig: figure dont les 3 premiers axes permettent de tracer, avg metrics, losses et dernières reward/action
    tous les autres axes permettent de tracer les dernières observations vues lors de l'evalution du modèle
    :param plot_obs: fonction(obs,ax) permettant de tracer une observation sur ax
    :param slice_: choix des dernières reward/action à tracer parmis les observations menées lors des évaluations du modèle
    :return:
    """
    def update_plot(step,trainer):

        for ax in fig.axes:
            ax.clear()

        ax,ax2,ax3 = fig.axes[:3]
        axes = fig.axes[3:]

        ax.plot(trainer.metrics,label="metric",color="green")
        losses = trainer.losses
        if losses.ndim==2:
            losses = losses.reshape((-1,2))
            ax2.plot(losses[:,0],label="actor_loss",color="red")
            ax2.plot(losses[:,1],label="critic_loss",color="green")
        else:
            ax2.plot(losses,label="loss",color="red")

        trainer.observer.plot_reward(ax=ax3,slice_=slice_)
        trainer.observer.plot_action(ax=ax3,slice_=slice_)
        if plot_obs is not None:
            obs = trainer.observer.observations()[-len(axes):]
            for o,ax_ in zip(obs,axes):
                plot_obs(o,ax_)
        if plot_traj is not None:
            trajs = trainer.observer.trajectories()[-len(axes):]
            for tr,ax_ in zip(trajs,axes):
                plot_traj(tr,ax_)
        ax.legend()
        ax2.legend()
    return update_plot


def fig_trainer(r,c,**kwargs):
    assert r%3==0
    fig = plt.figure(**kwargs)
    gs = GridSpec(r, c,width_ratios=[1]+[1/(c-1)]*(c-1))
    ax = fig.add_subplot(gs[0:r//3,0])
    ax2 = fig.add_subplot(gs[r//3:r//3*2,0])
    ax3 = fig.add_subplot(gs[r//3*2:,0])
    axes =[fig.add_subplot(gs[i,j+1]) for i,j in itertools.product(range(r),range(c-1))]
    return fig

#
# def video_trainer(*args):
#     results = []
#     for a in args:
#         assert isinstance(a,Figure)
#         results.append([])
#         Figure.arr
#     def callback(step,trainer):
#         for i,a in enumerate(args):



