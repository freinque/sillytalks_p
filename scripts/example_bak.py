# Derived from keras-rl
import opensim as osim
import numpy as np
import sys

from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, Input, concatenate, Lambda
from keras.optimizers import Adam

import numpy as np

from rl.agents import DDPGAgent
from rl.memory import SequentialMemory
from rl.random import OrnsteinUhlenbeckProcess

from osim.env import *
from osim.http.client import Client

from keras.optimizers import RMSprop

import argparse
import math

import sillywalks.controllers
import tensorflow as tf

#dcrease theta

DIFFICULTY = 0

# Command line parameters
parser = argparse.ArgumentParser(description='Train or test neural net motor controller')
parser.add_argument('--train', dest='train', action='store_true', default=True)
parser.add_argument('--test', dest='train', action='store_false', default=True)
parser.add_argument('--steps', dest='steps', action='store', default=10000, type=int)
parser.add_argument('--visualize', dest='visualize', action='store_true', default=False)
parser.add_argument('--model', dest='model', action='store', default="example.h5f")
parser.add_argument('--token', dest='token', action='store', required=False)
args = parser.parse_args()
# Total number of steps in training
nallsteps = args.steps

# Load walking environment
env = RunEnv(args.visualize)
state = env.reset(difficulty = DIFFICULTY, seed = None) 
print 'initial state: ', state

nb_actions = env.action_space.shape[0]
print 'nb_actions', nb_actions




# actor: state to action 
#actor = Input(shape=(sillywalks.controllers.DIM_STATE,), name='state_input')
actor = Sequential()

#model.add(Input(tensor=tf_embedding_input)) 
#model.add(Embedding(max_features, 128, input_length=maxlen))
reduce_state_layer = Dense(sillywalks.controllers.DIM_RED_STATE, trainable=False, kernel_initializer='zeros', input_shape=(1,)+env.observation_space.shape)
actor.add( reduce_state_layer )
#actor.add(Lambda(lambda x: sillywalks.controllers.reduce_state_tf(x), input_shape=(1,)+env.observation_space.shape ) )
#actor.add(Lambda(lambda x: tf.Tensor([x[22]-x[24],x[6],x[7],x[8],x[9],x[10],x[11],x[19],x[20]]), input_shape=(1,)+env.observation_space.shape ) )
#actor.add(Lambda(lambda x: tf.concat([x[:15], x[0:3]], axis=0 ), input_shape=(1,)+env.observation_space.shape ) )
#actor.add(Lambda(sillywalks.controllers.reduce_state_tf, input_shape=(1,)+env.observation_space.shape ) )
actor.add(Flatten())
actor.add(Dense(16))
actor.add(Activation('relu'))
actor.add(Dense(sillywalks.controllers.DIM_RED_ACTION))
actor.add(Activation('relu'))

inverse_reduce_action_layer = Dense(sillywalks.controllers.DIM_ACTION, trainable=False, kernel_initializer='zeros', input_shape=(1,)+(sillywalks.controllers.DIM_RED_ACTION,) )
actor.add( inverse_reduce_action_layer )

actor.add(Activation('sigmoid'))
print 'actor summary'
print(actor.summary())


ar_w = np.zeros(shape=(sillywalks.controllers.DIM_STATE, sillywalks.controllers.DIM_RED_STATE))
ar_w[22][0] = 1.
ar_w[24][0] = -1.
ar_w[6][1] = 1.
ar_w[7][2] = 1.
ar_w[8][3] = 1.
ar_w[9][4] = 1.
ar_w[10][5] = 1.
ar_w[11][6] = 1.
ar_w[19][7] = 1.
ar_w[20][8] = 1.
ar_b = np.zeros(shape=(sillywalks.controllers.DIM_RED_STATE))
weights = reduce_state_layer.set_weights([ar_w, ar_b])
print 'reduce_state_layer ', reduce_state_layer.get_weights()

ar_w = np.zeros(shape=(sillywalks.controllers.DIM_RED_ACTION, sillywalks.controllers.DIM_ACTION))
print ar_w
ar_w[0][2] = -1.
ar_w[0][4] = 1.
ar_w[1][6] = 1.
ar_w[1][7] = 1.
ar_w[1][8] = -1.
ar_w[2][6+9] = 1.
ar_w[2][7+9] = 1.
ar_w[2][8+9] = -1.
ar_w[3][1] = -1.
ar_w[3][5] = 1.
ar_w[3][1+9] = -1.
ar_w[3][5+9] = 1.
ar_w[5][0] = 1.
ar_w[5][3] = -1.
ar_w[5][0+9] = 1.
ar_w[5][3+9] = -1.
ar_b = np.zeros(shape=(sillywalks.controllers.DIM_ACTION))
weights = inverse_reduce_action_layer.set_weights([ar_w, ar_b])
print 'get inverse_reduce_action_layer ', inverse_reduce_action_layer.get_weights()



action_input = Input(shape=(nb_actions,), name='action_input')
observation_input = Input(shape=(1,) + env.observation_space.shape, name='observation_input')
flattened_observation = Flatten()(observation_input)

# critic: Q function
x = concatenate([action_input, flattened_observation])
x = Dense(64)(x)
x = Activation('relu')(x)
x = Dense(64)(x)
x = Activation('relu')(x)
x = Dense(64)(x)
x = Activation('relu')(x)
x = Dense(1)(x)
x = Activation('linear')(x)
critic = Model(inputs=[action_input, observation_input], outputs=x)
print 'critic summary'
print(critic.summary())

# Set up the agent for training
memory = SequentialMemory(limit=100000, window_length=1)
random_process = OrnsteinUhlenbeckProcess(theta=.015, mu=0., sigma=.2, size=env.noutput)
agent = DDPGAgent(nb_actions=nb_actions, actor=actor, critic=critic, critic_action_input=action_input,
                  memory=memory, nb_steps_warmup_critic=100, nb_steps_warmup_actor=100,
                  random_process=random_process, gamma=.99, target_model_update=1e-3,
                  delta_clip=1.)
# agent = ContinuousDQNAgent(nb_actions=env.noutput, V_model=V_model, L_model=L_model, mu_model=mu_model,
#                            memory=memory, nb_steps_warmup=1000, random_process=random_process,
#                            gamma=.99, target_model_update=0.1)
agent.compile(Adam(lr=.001, clipnorm=1.), metrics=['mae'])

# Okay, now it's time to learn something! We visualize the training here for show, but this
# slows down training quite a lot. You can always safely abort the training prematurely using
# Ctrl + C.
if args.train:
    agent.fit(env, nb_steps=nallsteps, visualize=False, verbose=1, nb_max_episode_steps=env.timestep_limit, log_interval=10000)
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


