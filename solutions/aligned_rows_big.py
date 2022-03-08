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



SOLUTION_NAME = "Aligned_Row_Big"


# ENVIRONNEMENT
environment = envs.AlignedRowEnv(16,action_float=False,
                                 fail_on_same=True,
                                 observation_1D = True,
                                 rewards = {
                                     "already_filled":-10,
                                     "max_iter":-10,
                                     "new_value":1,
                                     "success":10
                                 })
utils.validate_py_environment(environment, episodes=5)
train_env = tf_py_environment.TFPyEnvironment(environment)

# NETWORK
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


# AGENT
agent = DqnAgent(
    train_env.time_step_spec(),
    train_env.action_spec(),
    q_network=network((40,40),environment.dimension),
    optimizer=Adam(learning_rate=0.0005),
    td_errors_loss_fn=common.element_wise_squared_loss,)
agent.initialize()


trainer = trainer_iarch.Trainer(train_env,agent)
trainer.initialize_buffer(min_size=640,random_policy=True)




def plot_obs(obs,ax):
    return ax.imshow(obs,vmin=0.0,vmax=1.0)
fig_tr = fig_trainer(6,6,figsize=(20,20))

callbacks = [update_plotter(fig_tr,plot_obs),
             results_saver(10,SOLUTION_NAME,fig_tr),
             trainer_saver(10,SOLUTION_NAME)
             ]

trainer.run(callbacks=callbacks,
            buffer_size_increase_per_iteration = 100,
            sample_batch_size_experience = 64,
            num_iterations_train = 100,
            num_steps_per_row_in_experience = 2
            )

