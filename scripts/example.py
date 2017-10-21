# Derived from keras-rl
import opensim as osim
from osim.http.client import Client
import numpy as np
import sys
import argparse
import math

import tensorflow as tf
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, Input, concatenate, Lambda
from keras.optimizers import Adam, RMSprop

from rl.agents import DDPGAgent
from rl.memory import SequentialMemory
from rl.random import OrnsteinUhlenbeckProcess

import sillywalks.controllers
import sillywalks.env


############################################################################################

DIFFICULTY = 0

# Command line parameters
parser = argparse.ArgumentParser(description='Train or test neural net motor controller')
parser.add_argument('--train', dest='train', action='store_true', default=True)
parser.add_argument('--test', dest='train', action='store_false', default=True)
parser.add_argument('--steps', dest='steps', action='store', default=10000, type=int)
parser.add_argument('--visualize', dest='visualize', action='store_true', default=False)
parser.add_argument('--model', dest='model', action='store', default="example.h5f")
parser.add_argument('--loadweights', dest='loadweights', action='store_true', default=False)
parser.add_argument('--token', dest='token', action='store', required=False)
args = parser.parse_args()
# Total number of steps in training
nallsteps = args.steps

# Load walking environment
env = sillywalks.env.RunEnv(args.visualize)
state = env.reset(difficulty = DIFFICULTY, seed = None) 
print 'initial state: ', state

nb_actions = env.action_space.shape[0]
print 'nb_actions', nb_actions




# actor: state to action, i.e. policy
actor = Sequential()
reduce_state_layer = Dense(sillywalks.controllers.DIM_RED_STATE, trainable=False, kernel_initializer='zeros', input_shape=(1,)+env.observation_space.shape)
actor.add( reduce_state_layer )
actor.add(Flatten())

prop_control_layer = Dense(sillywalks.controllers.DIM_RED_ACTION, trainable=True, kernel_initializer='zeros')
actor.add(prop_control_layer)

actor.add(Dense(16))
actor.add(Activation('relu'))
actor.add(Dense(sillywalks.controllers.DIM_RED_ACTION))
actor.add(Activation('relu'))

inverse_reduce_action_layer = Dense(sillywalks.controllers.DIM_ACTION, trainable=False, kernel_initializer='zeros', input_shape=(1,)+(sillywalks.controllers.DIM_RED_ACTION,) )
actor.add( inverse_reduce_action_layer )

#actor.add(Activation('sigmoid'))
print 'actor summary'
print(actor.summary())

    ## set actor weights
weights = reduce_state_layer.set_weights([sillywalks.controllers.REDUCE_STATE_W, sillywalks.controllers.REDUCE_STATE_B])
print 'reduce_state_layer ', reduce_state_layer.get_weights()
weights = inverse_reduce_action_layer.set_weights([sillywalks.controllers.INVERSE_REDUCE_ACTION_W, sillywalks.controllers.INVERSE_REDUCE_ACTION_B])
print 'get inverse_reduce_action_layer ', inverse_reduce_action_layer.get_weights()

weights = prop_control_layer.set_weights([sillywalks.controllers.PROP_CONTROL_W, sillywalks.controllers.PROP_CONTROL_B])
print 'get prop_control_layer ', prop_control_layer.get_weights()



# critic: Q function
critic = Sequential()

action_input = Input(shape=(sillywalks.controllers.DIM_ACTION,), name='action_input')
observation_input = Input(shape=(1,) + (sillywalks.controllers.DIM_STATE,), name='observation_input')
flattened_observation = Flatten()(observation_input)

red_action_layer = Dense(sillywalks.controllers.DIM_RED_ACTION)
red_action = red_action_layer(action_input)
red_state_layer = Dense(sillywalks.controllers.DIM_RED_STATE)
red_state = red_state_layer(flattened_observation)

x = concatenate([red_action, red_state])
x = Dense(16)(x)
x = Activation('relu')(x)
x = Dense(16)(x)
x = Activation('relu')(x)
x = Dense(1)(x)
x = Activation('linear')(x)

critic = Model(inputs=[action_input, observation_input], outputs=x)
print 'critic summary'
print(critic.summary())

weights = red_state_layer.set_weights([sillywalks.controllers.REDUCE_STATE_W, sillywalks.controllers.REDUCE_STATE_B])
print 'reduce_state_layer ', red_state_layer.get_weights()

weights = red_action_layer.set_weights([sillywalks.controllers.REDUCE_ACTION_W, sillywalks.controllers.REDUCE_ACTION_B])
print 'get reduce_action_layer ', red_action_layer.get_weights()



# Set up the agent for training
memory = SequentialMemory(limit=100000, window_length=1)
random_process = OrnsteinUhlenbeckProcess(theta=.1, mu=0., sigma=.2, size=env.noutput)
agent = DDPGAgent(nb_actions=nb_actions, actor=actor, critic=critic, critic_action_input=action_input,
                  memory=memory, nb_steps_warmup_critic=100, nb_steps_warmup_actor=100,
                  random_process=random_process, gamma=.99, target_model_update=1e-3,
                  delta_clip=1.)
# agent = ContinuousDQNAgent(nb_actions=env.noutput, V_model=V_model, L_model=L_model, mu_model=mu_model,
#                            memory=memory, nb_steps_warmup=1000, random_process=random_process,
#                            gamma=.99, target_model_update=0.1)
agent.compile(Adam(lr=.001, clipnorm=1.), metrics=['mae'])


if args.loadweights:
    agent.load_weights(args.model)


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
    agent.test(env, nb_episodes=1, visualize=True, nb_max_episode_steps=500)


