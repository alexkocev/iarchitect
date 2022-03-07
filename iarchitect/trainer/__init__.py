import numpy as np
from tf_agents.metrics import py_metrics,tf_metrics
from tf_agents.drivers import py_driver,dynamic_episode_driver,dynamic_step_driver
from tf_agents.replay_buffers import tf_uniform_replay_buffer,py_uniform_replay_buffer
from tf_agents.policies import random_tf_policy
from tf_agents.trajectories import trajectory

from iarchitect.common.observer import ObserverTrajectory


class Trainer:
    def __init__(self,tf_env,agent,max_length_buffer=10000,loss_getter=None):
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
        self._loss_getter = loss_getter
        if self._loss_getter is None:
            self._loss_getter = self.default_loss_getter

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


    def initialize_buffer(self,policy=None,min_size=100,random_policy=False):
        assert (policy is None and random_policy) or (not random_policy)
        self.replay_buffer.clear()
        if random_policy:
            policy = random_tf_policy.RandomTFPolicy(self.tf_env.time_step_spec(),
                                                    self.tf_env.action_spec())
        if policy is None:
            policy = self.agent.collect_policy
        for _ in range(min_size):
            self.collect_step(policy)

    def collect_step(self,policy=None):
        if policy is None:
            policy = self.agent.collect_policy
        time_step = self.tf_env.current_time_step()
        action_step = policy.action(time_step)
        next_time_step = self.tf_env.step(action_step.action)
        traj = trajectory.from_transition(time_step, action_step, next_time_step)
        self.replay_buffer.add_batch(traj)

    def collect_training_data(self,num_steps_driver,num_episodes_driver):
        """
        Pour cette phase utilisation de collect_policy de self.agent
        :param num_steps_driver: sur ce nombre de steps un batch est ajouté à self.replay_buffer
        :return:
        """
        # self.replay_buffer.clear()

        policy = self.agent.collect_policy
        # policy = random_tf_policy.RandomTFPolicy(time_step_spec=self.tf_env.time_step_spec(),
        #                                          action_spec=self.tf_env.action_spec())

        if num_steps_driver is not None:
            dynamic_step_driver.DynamicStepDriver(
                self.tf_env,
                policy,
                observers=[self.replay_buffer.add_batch],
                num_steps=num_steps_driver).run()
        elif num_episodes_driver is not None:
            dynamic_episode_driver.DynamicEpisodeDriver(
                self.tf_env,
                policy,
                observers=[self.replay_buffer.add_batch],
                num_episodes=num_episodes_driver).run()
        else:
            raise Exception("au moins num_steps_driver or num_episodes_driver doit être None")



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
            # print(_,__,experience,"------------------------------------------")
            new_losses = self.agent.train(experience=experience)
            losses.append(self._loss_getter(new_losses))
        return losses


    def default_loss_getter(self,new_losses):
        return new_losses.loss


    # TODO AJOUTER UN MOYEN DE TERMINER CETTE BOUCLE
    def run(self,maximum_iterations=1000,
            buffer_size_increase_per_iteration=10,
            sample_batch_size_experience=64,
            num_iterations_train = 10,
            num_steps_per_row_in_experience = 2,
            num_episodes_evaluate_driver=10,
            callbacks = []):

        # assert bool(num_episodes_collect_driver is None) ^ bool(num_steps_collect_driver is None) , "Choisir entre collect data by episodes num_steps_collect_driver=None) ou by steps (num_episodes_collect_driver = None)"
        abort = False
        for i in range(maximum_iterations):
            try:
                for _ in range(buffer_size_increase_per_iteration):
                    self.collect_step()
                new_losses = self.train_agent(
                    sample_batch_size=sample_batch_size_experience,
                    num_iterations = num_iterations_train,
                    num_steps=num_steps_per_row_in_experience)
                self.losses = np.append(self.losses,
                                        new_losses)
                self.metrics = np.append(self.metrics,
                                         [self.evaluate_agent(num_episodes_driver=num_episodes_evaluate_driver)]*len(new_losses))
            except KeyboardInterrupt:
                abort = True
            finally:
                for callback in callbacks:
                    try:
                        callback(i,self)
                    except KeyboardInterrupt:
                        pass
            if abort:
                break
