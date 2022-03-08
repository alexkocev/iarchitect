import json
import sys
from argparse import ArgumentParser
from pathlib import Path

import numpy as np


from iarchitect import envs,trainer as trainer_iarch
from iarchitect.common.callbacks import output_updater,update_plotter,fig_trainer,results_saver,trainer_saver

from tf_agents.agents import DdpgAgent,DqnAgent

from tf_agents.utils import common
from tf_agents.networks import sequential
from tf_agents.environments import tf_py_environment
from tf_agents.environments import utils
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import initializers

from iarchitect.solutions.common import ValidEnv, write_args


def make_environment(args):
    env = envs.WindowEnv(args.dimension,
                     np.fromiter(range(10),dtype=int),
                     render_dims=(args.dimension**0.5,args.dimension**0.5),
                     random_reset=args.random_reset,
                     max_species_reset = args.max_species_reset,
                     strategie=args.strategie,
                     discount=args.discount)
    utils.validate_py_environment(env, episodes=5)
    tf_env = tf_py_environment.TFPyEnvironment(env)
    return env,tf_env

if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument("--random_reset",default=1,choices=("0","1"))
    parser.add_argument("--max_species_reset",default=None)
    parser.add_argument("--strategie",default=1,type=int)
    parser.add_argument("--dimension",default=16,type=int)
    parser.add_argument("--discount",default=1,type=float)
    parser.add_argument("--greedy_epsilon",default=0.1,type=float)
    parser.add_argument("--suff",default=None)
    parser.add_argument("--maximum_iterations",default=1000,type=int)
    parser.add_argument("--num_iterations_train",default=10,type=int)
    parser.add_argument("--sample_batch_size_experience",default=64,type=int)


    args = parser.parse_args(sys.argv[1:])
    args.random_reset = bool(int(args.random_reset))
    args.max_species_reset = args.max_species_reset if args.max_species_reset is None else int(args.max_species_reset)
    setattr(args,"name_env",ValidEnv.windows_env_big_network)
    print(args)
    assert args.dimension == int(args.dimension**0.5)**2

    SOLUTION_NAME = f"WindowEnvWhatPlantGrosReseau_discount{str(args.discount).replace('.','_')}_dimension{args.dimension}_reset{args.random_reset}_maxspe{args.max_species_reset}_str{args.strategie}"
    if args.suff is not None:
        SOLUTION_NAME += "_" + args.suff

    p = Path(SOLUTION_NAME)
    p.mkdir(exist_ok=False)
    write_args(p,args)

    environment,train_env = make_environment(args)


    def dense_layer(num_units):
        return layers.Dense(
            num_units,
            activation="relu",
            kernel_initializer=initializers.VarianceScaling(
                scale=2.0, mode='fan_in', distribution='truncated_normal'))

    def network(fc_layers_units,dimension_q_values):
        q_values_layer = layers.Dense(
            dimension_q_values,
            activation=None,
            kernel_initializer=initializers.RandomUniform(
                minval=-0.03, maxval=0.03),
            bias_initializer=initializers.Constant(-0.2))
        return sequential.Sequential([layers.Flatten()] + [dense_layer(n) for n in fc_layers_units] + [q_values_layer])


    agent = DqnAgent(
        train_env.time_step_spec(),
        train_env.action_spec(),
        q_network=network((64,64),len(environment.tuiles)),
        optimizer=Adam(learning_rate=0.0005),
        td_errors_loss_fn=common.element_wise_squared_loss,
        epsilon_greedy = args.greedy_epsilon)
    agent.initialize()



    def plot_obs(obs,ax):
        return ax.imshow(obs,vmin=0.0,vmax=1.0)

    def plot_traj(tr,ax):
        return ax.imshow(tr)



    trainer = trainer_iarch.Trainer(train_env,agent)
    trainer.initialize_buffer(min_size=640,random_policy=True)


    fig_tr = fig_trainer(6,6,figsize=(20,20))
    callbacks = [update_plotter(fig_tr,plot_obs),
                 results_saver(10,SOLUTION_NAME,fig_tr),
                 trainer_saver(10,SOLUTION_NAME)
                 ]

    trainer.run(callbacks=callbacks,
                buffer_size_increase_per_iteration = 10,
                sample_batch_size_experience = args.sample_batch_size_experience,
                num_steps_per_row_in_experience = 2,
                maximum_iterations=args.maximum_iterations,
                num_iterations_train = args.num_iterations_train
                )

