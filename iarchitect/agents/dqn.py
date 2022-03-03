from tf_agents.agents import DqnAgent

from tf_agents.utils import common
from tf_agents.networks import sequential

from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import initializers


def dense_layer(num_units):
    return layers.Dense(
        num_units,
        activation="relu",
        kernel_initializer=initializers.VarianceScaling(
            scale=2.0, mode='fan_in', distribution='truncated_normal'))

def network_factory(fc_layers_units=(4, 4),dimension_q_values=4):
    q_values_layer = layers.Dense(
        dimension_q_values,
        activation=None,
        kernel_initializer=initializers.RandomUniform(
            minval=-0.03, maxval=0.03),
        bias_initializer=initializers.Constant(-0.2))
    q_net = sequential.Sequential([dense_layer(n) for n in fc_layers_units] + [q_values_layer])
    return q_net

def agent_factory(tf_env,
                  network=network_factory(),
                  learning_rate=0.0005):
    agent = DqnAgent(
        tf_env.time_step_spec(),
        tf_env.action_spec(),
        q_network=network,
        optimizer=Adam(learning_rate=learning_rate),
        td_errors_loss_fn=common.element_wise_squared_loss,)
    agent.initialize()
    return agent
