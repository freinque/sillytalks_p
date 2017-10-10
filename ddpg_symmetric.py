from __future__ import division
from collections import deque
import os
import warnings
import copy

import numpy as np
import keras.backend as K
import keras.optimizers as optimizers
from keras.layers import Lambda
import tensorflow as tf

from rl.core import Agent, Processor
from rl.random import OrnsteinUhlenbeckProcess
from rl.util import *

K.set_learning_phase(1) #set learning phase

def mean_q(y_true, y_pred):
    return K.mean(K.max(y_pred, axis=-1))


def slice_action_right(x):
    '''Take the first half of the action vector, corresponding to the right side of the body
    '''
    return x[:, :9]


def slice_action_left(x):
    '''Take the second half of the action vector, corresponding to the left side of the body
    '''
    return x[:, 9:]


def slice_obs_right(x):
    '''Take the portion of the observation vector corresponding to the right side of the body
    '''
    return x[np.array([6, 7, 8,  # rotation of each ankle, knee and hip
                       12, 13, 14,  # angular velocity of each ankle, knee and hip
                       30, 31,  # right toes
                       34, 35,  # right talus
                       ])]


def slice_obs_left(x):
    '''Take the portion of the observation vector corresponding to the left side of the body
    '''
    return x[np.array([9, 10, 11,  # rotation of each ankle, knee and hip
                       15, 16, 17,  # angular velocity of each ankle, knee and hip
                       28, 29,  # left toes
                       32, 33,  # left talus
                       ])]


def slice_obs_the_rest(x):
    '''Take the portion of the observation vector that is not related to one side of the body or the other.
    Note that we use this to drop the redundant 'position of the pelvis' coords
    '''
    return x[np.array([0, 1, 2,  # pelvis position
                       3, 4, 5,  # pelvis velocity
                       18, 19,  # position of the center of mass (2 values)
                       20, 21,  # velocity of the center of mass (2 values)
                       22, 23,  # positions (x, y) of head
                       # 24,25, #pelvis
                       26, 27,  # torso
                       36, 37,
                       # strength of left and right psoas (not putting this with legs because it's where the legs connect to the rest)
                       38, 39, 40  # obstacle
                       ])]


class SymmetricProcessor(Processor):
    def __init__(self):
        super(SymmetricProcessor, self).__init__()

    def process_observation(self, observation):
        return observation

    def process_state_batch(self, batch):
        right = np.array([slice_obs_right(c[0]) for c in batch])
        left = np.array([slice_obs_left(c[0]) for c in batch])
        the_rest = np.array([slice_obs_the_rest(c[0]) for c in batch])
        return [right, left, the_rest]


# Deep DPG as described by Lillicrap et al. (2015)
# http://arxiv.org/pdf/1509.02971v2.pdf
# http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.646.4324&rep=rep1&type=pdf
class DDPGSymmetricAgent(Agent):
    """Modified version of DPPGAgent, takes multiple inputs.  Currently hard-coded for our specific application.
    """
    def __init__(self, nb_actions, actor, critic,
                 actor_inputs,
                 critic_inputs,
                 memory,
                 gamma=.99, batch_size=32, nb_steps_warmup_critic=1000, nb_steps_warmup_actor=1000,
                 train_interval=1, memory_interval=1, delta_range=None, delta_clip=np.inf,
                 random_process=None, custom_model_objects={}, target_model_update=.001, **kwargs):
        if hasattr(actor.output, '__len__') and len(actor.output) > 1:
            raise ValueError('Actor "{}" has more than one output. DDPG expects an actor that has a single output.'.format(actor))
        if hasattr(critic.output, '__len__') and len(critic.output) > 1:
            raise ValueError('Critic "{}" has more than one output. DDPG expects a critic that has a single output.'.format(critic))
        #if critic_action_input not in critic.input:
        #    raise ValueError('Critic "{}" does not have designated action input "{}".'.format(critic, critic_action_input))
        if not hasattr(critic.input, '__len__') or len(critic.input) < 2:
            raise ValueError('Critic "{}" does not have enough inputs. The critic must have at exactly two inputs, one for the action and one for the observation.'.format(critic))

        super(DDPGSymmetricAgent, self).__init__(**kwargs)

        # Soft vs hard target model updates.
        if target_model_update < 0:
            raise ValueError('`target_model_update` must be >= 0.')
        elif target_model_update >= 1:
            # Hard update every `target_model_update` steps.
            target_model_update = int(target_model_update)
        else:
            # Soft update with `(1 - target_model_update) * old + target_model_update * new`.
            target_model_update = float(target_model_update)

        if delta_range is not None:
            warnings.warn('`delta_range` is deprecated. Please use `delta_clip` instead, which takes a single scalar. For now we\'re falling back to `delta_range[1] = {}`'.format(delta_range[1]))
            delta_clip = delta_range[1]

        # Parameters.
        self.nb_actions = nb_actions
        self.nb_steps_warmup_actor = nb_steps_warmup_actor
        self.nb_steps_warmup_critic = nb_steps_warmup_critic
        self.random_process = random_process
        self.delta_clip = delta_clip
        self.gamma = gamma
        self.target_model_update = target_model_update
        self.batch_size = batch_size
        self.train_interval = train_interval
        self.memory_interval = memory_interval
        self.custom_model_objects = custom_model_objects

        # Related objects.
        self.actor = actor
        self.critic = critic
        self.actor_inputs = actor_inputs
        self.critic_inputs = critic_inputs
        self.memory = memory

        # State.
        self.compiled = False
        self.reset_states()

    @property
    def uses_learning_phase(self):
        return self.actor.uses_learning_phase or self.critic.uses_learning_phase

    def compile(self, optimizer, metrics=[]):
        metrics += [mean_q]

        if type(optimizer) in (list, tuple):
            if len(optimizer) != 2:
                raise ValueError('More than two optimizers provided. Please only provide a maximum of two optimizers, the first one for the actor and the second one for the critic.')
            actor_optimizer, critic_optimizer = optimizer
        else:
            actor_optimizer = optimizer
            critic_optimizer = clone_optimizer(optimizer)
        if type(actor_optimizer) is str:
            actor_optimizer = optimizers.get(actor_optimizer)
        if type(critic_optimizer) is str:
            critic_optimizer = optimizers.get(critic_optimizer)
        assert actor_optimizer != critic_optimizer

        if len(metrics) == 2 and hasattr(metrics[0], '__len__') and hasattr(metrics[1], '__len__'):
            actor_metrics, critic_metrics = metrics
        else:
            actor_metrics = critic_metrics = metrics

        def clipped_error(y_true, y_pred):
            return K.mean(huber_loss(y_true, y_pred, self.delta_clip), axis=-1)

        # Compile target networks. We only use them in feed-forward mode, hence we can pass any
        # optimizer and loss since we never use it anyway.
        self.target_actor = clone_model(self.actor, self.custom_model_objects)
        self.target_actor.compile(optimizer='sgd', loss='mse')
        self.target_critic = clone_model(self.critic, self.custom_model_objects)
        self.target_critic.compile(optimizer='sgd', loss='mse')

        # We also compile the actor. We never optimize the actor using Keras but instead compute
        # the policy gradient ourselves. However, we need the actor in feed-forward mode, hence
        # we also compile it with any optimzer and
        self.actor.compile(optimizer='sgd', loss='mse')

        # Compile the critic.
        if self.target_model_update < 1.:
            # We use the `AdditionalUpdatesOptimizer` to efficiently soft-update the target model.
            critic_updates = get_soft_target_model_updates(self.target_critic, self.critic, self.target_model_update)
            critic_optimizer = AdditionalUpdatesOptimizer(critic_optimizer, critic_updates)
        self.critic.compile(optimizer=critic_optimizer, loss=clipped_error, metrics=critic_metrics)


        # critic takes [critic_right_action_input, critic_left_action_input,
        #               critic_right_obs_input, critic_left_obs_input, critic_the_rest_input]

        # actor takes [actor_right_obs_input, actor_left_obs_input, actor_the_rest_input]

        action = self.actor(self.actor_inputs)

        critic_inputs = [
            Lambda(slice_action_right, output_shape=(9,), name='critic_right_action_input')(action),
            Lambda(slice_action_left, output_shape=(9,), name='critic_left_action_input')(action),
            self.critic.inputs[2], self.critic_inputs[3], self.critic_inputs[4]
        ]

        critic_output = self.critic(critic_inputs)

        updates = actor_optimizer.get_updates(self.actor.trainable_weights, self.actor.constraints,
                                              loss=-K.mean(critic_output))
        if self.target_model_update < 1.:
            # Include soft target model updates.
            updates += get_soft_target_model_updates(self.target_actor, self.actor, self.target_model_update)
        updates += self.actor.updates  # include other updates of the actor, e.g. for BN

        # Finally, combine it all into a callable function.
        actor_inputs = self.actor_inputs
        if self.uses_learning_phase:
            actor_inputs = self.actor_inputs + [K.learning_phase()]
            #actor_inputs = self.actor_inputs + [tf.placeholder(dtype='bool', name='keras_learning_phase')]
        #import pdb; pdb.set_trace()
        self.actor_train_fn = K.function(actor_inputs, [self.actor(actor_inputs)], updates=updates)
        self.actor_optimizer = actor_optimizer

        self.compiled = True
        print('compiled successfully')

    def load_weights(self, filepath):
        filename, extension = os.path.splitext(filepath)
        actor_filepath = filename + '_actor' + extension
        critic_filepath = filename + '_critic' + extension
        self.actor.load_weights(actor_filepath)
        self.critic.load_weights(critic_filepath)
        self.update_target_models_hard()

    def save_weights(self, filepath, overwrite=False):
        filename, extension = os.path.splitext(filepath)
        actor_filepath = filename + '_actor' + extension
        critic_filepath = filename + '_critic' + extension
        self.actor.save_weights(actor_filepath, overwrite=overwrite)
        self.critic.save_weights(critic_filepath, overwrite=overwrite)

    def update_target_models_hard(self):
        self.target_critic.set_weights(self.critic.get_weights())
        self.target_actor.set_weights(self.actor.get_weights())

    # TODO: implement pickle

    def reset_states(self):
        if self.random_process is not None:
            self.random_process.reset_states()
        self.recent_action = None
        self.recent_observation = None
        if self.compiled:
            self.actor.reset_states()
            self.critic.reset_states()
            self.target_actor.reset_states()
            self.target_critic.reset_states()

    def process_state_batch(self, batch):
        '''input is list, e.g. [[[-0.05, 0.0, 0.91, 0.0]]]
        '''
        batch = np.array(batch) # shape is now (n_states, 1, len_state)
        if self.processor is None:
            return batch
        return self.processor.process_state_batch(batch)

    def select_action(self, state):
        '''state is a list of lists, e.g. [[],[]]

        '''
        batch = self.process_state_batch([state])
        action = self.actor.predict_on_batch(batch).flatten()
        assert action.shape == (self.nb_actions,)

        # Apply noise, if a random process is set.
        if self.training and self.random_process is not None:
            noise = self.random_process.sample()
            assert noise.shape == action.shape
            action += noise

        return action

    def forward(self, observation):
        # Select an action.
        state = self.memory.get_recent_state(observation)
        action = self.select_action(state)  # TODO: move this into policy
        if self.processor is not None:
            action = self.processor.process_action(action)

        # Book-keeping.
        self.recent_observation = observation
        self.recent_action = action

        return action

    @property
    def layers(self):
        return self.actor.layers[:] + self.critic.layers[:]

    @property
    def metrics_names(self):
        names = self.critic.metrics_names[:]
        if self.processor is not None:
            #import pdb; pdb.set_trace()
            names += self.processor.metrics_names[:]
        return names

    def backward(self, reward, terminal=False):
        # Store most recent experience in memory.
        if self.step % self.memory_interval == 0:
            self.memory.append(self.recent_observation, self.recent_action, reward, terminal,
                               training=self.training)

        metrics = [np.nan for _ in self.metrics_names]
        if not self.training:
            # We're done here. No need to update the experience memory since we only use the working
            # memory to obtain the state over the most recent observations.
            return metrics

        # Train the network on a single stochastic batch.
        can_train_either = self.step > self.nb_steps_warmup_critic or self.step > self.nb_steps_warmup_actor
        if can_train_either and self.step % self.train_interval == 0:
            experiences = self.memory.sample(self.batch_size)
            assert len(experiences) == self.batch_size

            # Start by extracting the necessary parameters (we use a vectorized implementation).
            state0_batch = []
            reward_batch = []
            action_batch = []
            terminal1_batch = []
            state1_batch = []
            for e in experiences:
                state0_batch.append(e.state0)
                state1_batch.append(e.state1)
                reward_batch.append(e.reward)
                action_batch.append(e.action)
                terminal1_batch.append(0. if e.terminal1 else 1.)

            # Prepare and validate parameters.
            #import pdb; pdb.set_trace()
            state0_batch = self.process_state_batch(state0_batch)
            state1_batch = self.process_state_batch(state1_batch)
            terminal1_batch = np.array(terminal1_batch)
            reward_batch = np.array(reward_batch)
            action_batch = np.array(action_batch)
            assert reward_batch.shape == (self.batch_size,)
            assert terminal1_batch.shape == reward_batch.shape
            assert action_batch.shape == (self.batch_size, self.nb_actions)

            # Update critic, if warm up is over.
            if self.step > self.nb_steps_warmup_critic:

                # critic takes [critic_right_action_input, critic_left_action_input,
                #               critic_right_obs_input, critic_left_obs_input, critic_the_rest_input]

                target_actions = self.target_actor.predict_on_batch(state1_batch)
                assert target_actions.shape == (self.batch_size, self.nb_actions)
                state1_batch_with_action = state1_batch
                target_actions_right = np.array([np.array(t[:9]) for t in target_actions])
                target_actions_left = np.array([np.array(t[9:]) for t in target_actions])
                state1_batch_with_action.insert(0, target_actions_left)
                state1_batch_with_action.insert(0, target_actions_right)

                target_q_values = self.target_critic.predict_on_batch(state1_batch_with_action).flatten()
                assert target_q_values.shape == (self.batch_size,)

                # Compute r_t + gamma * max_a Q(s_t+1, a) and update the target ys accordingly,
                # but only for the affected output units (as given by action_batch).
                discounted_reward_batch = self.gamma * target_q_values
                discounted_reward_batch *= terminal1_batch
                assert discounted_reward_batch.shape == reward_batch.shape
                targets = (reward_batch + discounted_reward_batch).reshape(self.batch_size, 1)

                # Perform a single batch update on the critic network.
                state0_batch_with_action = copy.deepcopy(state0_batch)

                actions_right = np.array([np.array(t[:9]) for t in action_batch])
                actions_left = np.array([np.array(t[9:]) for t in action_batch])
                state0_batch_with_action.insert(0, actions_left)
                state0_batch_with_action.insert(0, actions_right)
                metrics = self.critic.train_on_batch(state0_batch_with_action, targets)
                if self.processor is not None:
                    metrics += self.processor.metrics

            # Update actor, if warm up is over.
            if self.step > self.nb_steps_warmup_actor:
                # TODO: implement metrics for actor
                # actor takes [actor_right_obs_input, actor_left_obs_input, actor_the_rest_input]
                inputs = state0_batch
                if self.uses_learning_phase:
                    inputs += [self.training]
                action_values = self.actor_train_fn(inputs)[0]
                assert action_values.shape == (self.batch_size, self.nb_actions)

        if self.target_model_update >= 1 and self.step % self.target_model_update == 0:
            self.update_target_models_hard()

        return metrics

