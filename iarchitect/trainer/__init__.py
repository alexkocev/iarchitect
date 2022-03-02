import numpy as np
from tf_agents.metrics import py_metrics,tf_metrics
from tf_agents.drivers import py_driver,dynamic_episode_driver,dynamic_step_driver
from tf_agents.replay_buffers import tf_uniform_replay_buffer,py_uniform_replay_buffer

from iarchitect.common.observer import ObserverTrajectory


class Trainer:
    def __init__(self,tf_env,agent,max_length_buffer=10000):
        """

        :param tf_env: environment tensor_flow
        :param agent: agent à entrainer
        :param max_length_buffer: taille maximale du buffer pour son initialisation
        """
        self.tf_env = tf_env
        self.agent = agent
        self.observer = ObserverTrajectory(verbose=False)
        self.replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
            self.agent.collect_data_spec,
            batch_size=tf_env.batch_size,
            max_length=max_length_buffer)
        self.metrics = np.array([])
        self.losses = np.array([])


    def evaluate_agent(self,num_episodes_driver=100):
        """

        :param num_episodes_driver: la métrique moyenne est calculée sur ce nombre d'épisode
        :return:
        """
        metric = tf_metrics.AverageReturnMetric()
        observers = [metric,self.observer]
        driver = dynamic_episode_driver.DynamicEpisodeDriver(
            self.tf_env, self.agent.policy, observers, num_episodes=num_episodes_driver)
        final_time_step, policy_state = driver.run()
        return metric.result().numpy()

    def collect_training_data(self,num_steps_driver=1000):
        """
        Pour cette phase utilisation de collect_policy de self.agent
        :param num_steps_driver: sur ce nombre de steps un batch est ajouté à self.replay_buffer
        :return:
        """
        dynamic_step_driver.DynamicStepDriver(
            self.tf_env,
            self.agent.collect_policy,
            observers=[self.replay_buffer.add_batch],
            num_steps=num_steps_driver).run()

    def train_agent(self,sample_batch_size=64,num_steps=2,num_iterations=100):
        """

        :param sample_batch_size: taille d'un sample par expérience
        :param num_steps: longueur de chaque ligne dans un sample
        :param num_iterations: nombre d'experiences sur lequel est entrainé le modèle avant de sortir
        :return:
        """
        dataset = self.replay_buffer.as_dataset(
            sample_batch_size=sample_batch_size,
            num_steps=num_steps)
        iterator = iter(dataset)

        losses = []
        for _ in range(num_iterations):
            experience, __ = next(iterator)
            losses.append(self.agent.train(experience=experience).loss.numpy())
        return losses


    # TODO AJOUTER UN MOYEN DE TERMINER CETTE BOUCLE
    def run(self,maximum_iterations=1000,
            num_steps_collect_driver=64,
            sample_batch_size_experience=64,
            num_iterations_train = 10,
            num_steps_per_row_in_experience = 2,
            num_episodes_evaluate_driver=10,
            callbacks = []):
        for i in range(maximum_iterations):
            self.collect_training_data(
                num_steps_driver=num_steps_collect_driver
            )
            new_losses = self.train_agent(
                sample_batch_size=sample_batch_size_experience,
                num_iterations = num_iterations_train,
                num_steps=num_steps_per_row_in_experience)
            self.losses = np.append(self.losses,
                                    new_losses)
            self.metrics = np.append(self.metrics,
                                     [self.evaluate_agent(num_episodes_driver=num_episodes_evaluate_driver)]*len(new_losses))
            for callback in callbacks:
                callback(i,self)
