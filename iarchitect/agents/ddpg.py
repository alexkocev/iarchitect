import numpy as np
from tf_agents.agents import DdpgAgent
from tf_agents.utils import common
from tf_agents.networks import sequential
from tf_agents.agents import ddpg

from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import initializers



def agent_factory(tf_env,
                  actor_network = None,
                  critic_network = None,
                  learning_rate_actor=0.0005,
                  learning_rate_critic=0.0005,
                  kwargs_agent = {}
                  ):

    assert actor_network is not None and critic_network is not None
    # observation_spec, action_spec = tf_env.observation_spec(), tf_env.action_spec()
    # if critic_network is None:
    #     critic_network = ddpg.critic_network.CriticNetwork(
    #         (observation_spec, action_spec),
    #         observation_conv_layer_=[(tf_env.dimension*2,max(2,int(tf_env.dimension/10)),1)]*3,
    #         )
    #     # PAR DEFAUT 3 LAYER KERNEL = max(2,obs_dim/10)
    # if actor_network is None:
    #     actor_network = ddpg.actor_network.ActorNetwork(
    #         observation_spec, action_spec,
    #         fc_layer_params=[tuple(np.fromiter(tf_env.observation_spec().shape,dtype=int)*10)]*3)
    #     # PAR DEFAUT 3 LAYER DIX FOIS LA SHAPE DE L'OBSERVATION

    agent = DdpgAgent(
        tf_env.time_step_spec(),
        tf_env.action_spec(),
        actor_network,
        critic_network,
        actor_optimizer=Adam(learning_rate=learning_rate_actor),
        critic_optimizer=Adam(learning_rate=learning_rate_critic),
        **kwargs_agent)

#     actor_network: tf_agents.networks.Network,
#     critic_network: tf_agents.networks.Network,
#     actor_optimizer: Optional[types.Optimizer] = None,
#     critic_optimizer: Optional[types.Optimizer] = None,
#     ou_stddev: tf_agents.typing.types.Float = 1.0,
#     ou_damping: tf_agents.typing.types.Float = 1.0,
#     target_actor_network: Optional[tf_agents.networks.Network] = None,
#     target_critic_network: Optional[tf_agents.networks.Network] = None,
#     target_update_tau: tf_agents.typing.types.Float = 1.0,
#     target_update_period: tf_agents.typing.types.Int = 1,
#     dqda_clipping: Optional[types.Float] = None,
#     td_errors_loss_fn: Optional[tf_agents.typing.types.LossFn] = None,
#     gamma: tf_agents.typing.types.Float = 1.0,
#     reward_scale_factor: tf_agents.typing.types.Float = 1.0,
#     gradient_clipping: Optional[types.Float] = None,
#     debug_summaries: bool = False,
#     summarize_grads_and_vars: bool = False,
#     train_step_counter: Optional[tf.Variable] = None,
#     name: Optional[Text] = None
# )
    agent.initialize()
    return agent
