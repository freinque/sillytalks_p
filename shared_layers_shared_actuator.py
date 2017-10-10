# Derived from keras-rl
import opensim as osim
import numpy as np
import sys

from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, Input, concatenate, BatchNormalization, Lambda
from keras.optimizers import Adam
from keras.layers.core import K
from keras.initializers import Constant
import tensorflow as tf

import numpy as np

from rl.agents import DDPGAgent
from rl.memory import SequentialMemory
from rl.random import OrnsteinUhlenbeckProcess
from ddpg_symmetric import DDPGSymmetricAgent, SymmetricProcessor

from osim.env import *
from osim.http.client import Client

from keras.optimizers import RMSprop

import argparse
import math

# Command line parameters
parser = argparse.ArgumentParser(description='Train or test neural net motor controller')
parser.add_argument('--train', dest='train', action='store_true', default=True)
parser.add_argument('--test', dest='train', action='store_false', default=True)
parser.add_argument('--steps', dest='steps', action='store', default=10000, type=int)
parser.add_argument('--visualize', dest='visualize', action='store_true', default=False)
parser.add_argument('--model', dest='model', action='store', default="example.h5f")
parser.add_argument('--token', dest='token', action='store', required=False)
args = parser.parse_args()

# Load walking environment
env = RunEnv(args.visualize)
env.reset()

nb_actions = env.action_space.shape[0]

# Total number of steps in training
nallsteps = args.steps

# Create networks for DDPG

K.set_learning_phase(1) #set learning phase
K._LEARNING_PHASE = tf.constant(1) # try with 1


def slice_hidden_right(x):
    '''Take the first half of the hidden layer, which will control the right side of the body
    '''
    return x[:, :14]


def slice_hidden_left(x):
    '''Take the second half of the hidden layer, which will control the left side of the body
    '''
    return x[:, 14:]


## ACTOR
left_obs_input = Input(shape=(10,), name='left_obs_input')
right_obs_input = Input(shape=(10,), name='right_obs_input')
actor_shared_dense_obs = Dense(5)
actor_obs_encoded_left = actor_shared_dense_obs(left_obs_input)
actor_obs_encoded_right = actor_shared_dense_obs(right_obs_input)
the_rest_input = Input(shape=(19,), name='the_rest_input')

x = concatenate([actor_obs_encoded_right, actor_obs_encoded_left, the_rest_input])
x = Dense(29)(x)
x = Activation('relu')(x)
x = Dense(29)(x)
x = Activation('relu')(x)
x = Dense(28)(x)
x = Activation('relu')(x)
r = Lambda(slice_hidden_right, output_shape=(14,), name='right_actuator')(x)
l = Lambda(slice_hidden_left, output_shape=(14,), name='left_actuator')(x)
actuator_encoder = Dense(9, bias_initializer=Constant(value=-2))
encoded_right_actuator = actuator_encoder(r)
encoded_left_actuator = actuator_encoder(l)
x = concatenate([encoded_right_actuator, encoded_left_actuator])
x = Activation('sigmoid')(x)
actor = Model(inputs=[right_obs_input, left_obs_input, the_rest_input], outputs=x)
print(actor.summary())

## CRITIC

critic_shared_dense_obs = Dense(5)
critic_obs_encoded_left = critic_shared_dense_obs(left_obs_input)
critic_obs_encoded_right = critic_shared_dense_obs(right_obs_input)

critic_left_action_input = Input(shape=(9,), name='critic_left_action_input')
critic_right_action_input = Input(shape=(9,), name='critic_right_action_input')
critic_shared_dense_action = Dense(5)
critic_action_encoded_left = critic_shared_dense_action(critic_left_action_input)
critic_action_encoded_right = critic_shared_dense_action(critic_right_action_input)

x = concatenate([critic_action_encoded_right, critic_action_encoded_left,
                 critic_obs_encoded_right, critic_obs_encoded_left,
                 the_rest_input])
x = Dense(40)(x)
x = Activation('relu')(x)
x = Dense(40)(x)
#x = Activation('relu')(x)
#x = Dense(32)(x)
x = Activation('relu')(x)
x = Dense(1)(x)
x = Activation('linear')(x)
critic = Model(inputs=[critic_right_action_input, critic_left_action_input,
                       right_obs_input, left_obs_input, the_rest_input], outputs=x)
print(critic.summary())

# Set up the agent for training
memory = SequentialMemory(limit=100000, window_length=1)
random_process = OrnsteinUhlenbeckProcess(theta=.40, mu=0., sigma=.1, size=env.noutput)
agent = DDPGSymmetricAgent(nb_actions=nb_actions, actor=actor, critic=critic,
                           actor_inputs=[right_obs_input, left_obs_input, the_rest_input],
                           critic_inputs=[critic_right_action_input, critic_left_action_input,
                                          right_obs_input, left_obs_input, the_rest_input],
                           memory=memory, nb_steps_warmup_critic=100, nb_steps_warmup_actor=100,
                           random_process=random_process, gamma=.99, target_model_update=1e-3, train_interval=10, delta_clip=1.,
                           processor=SymmetricProcessor()
                           )
agent.compile(Adam(lr=.001, clipnorm=1.), metrics=['mae'])

# Okay, now it's time to learn something! We visualize the training here for show, but this
# slows down training quite a lot. You can always safely abort the training prematurely using
# Ctrl + C.
if args.train:
    agent.load_weights(args.model)
    agent.fit(env, nb_steps=nallsteps, visualize=False, verbose=1, nb_max_episode_steps=env.timestep_limit,
              log_interval=10000, action_repetition=2)
    # After training is done, we save the final weights.
    agent.save_weights(args.model, overwrite=True)

# If TEST and TOKEN, submit to crowdAI
if not args.train and args.token:
    agent.load_weights(args.model)
    # Settings
    remote_base = 'http://grader.crowdai.org:1729'
    client = Client(remote_base)

    # Create environment
    observation = client.env_create(args.token)

    # Run a single step
    # The grader runs 3 simulations of at most 1000 steps each. We stop after the last one
    while True:
        v = np.array(observation).reshape((env.observation_space.shape[0]))
        action = agent.forward(v)
        [observation, reward, done, info] = client.env_step(action.tolist())
        if done:
            observation = client.env_reset()
            if not observation:
                break

    client.submit()

# If TEST and no TOKEN, run some test experiments
if not args.train and not args.token:
    agent.load_weights(args.model)
    # Finally, evaluate our algorithm for 1 episode.
    agent.test(env, nb_episodes=1, visualize=False, nb_max_episode_steps=500)

