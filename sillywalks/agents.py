import tensorflow as tf
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, Input, concatenate, Lambda
from keras.optimizers import Adam, RMSprop

from rl.agents import DDPGAgent
from rl.memory import SequentialMemory
from rl.random import OrnsteinUhlenbeckProcess

import sillywalks.params

##########################################################################################################
################################## actor: state to action ################################################
    ## input is state
state_input = Input(shape=(1,) + (sillywalks.params.DIM_STATE,), name='state_input')
flattened_state_input = Flatten()(state_input)

    ## go through reduction
red_state_layer = Dense(sillywalks.params.DIM_RED_STATE, trainable=False, kernel_initializer='zeros')
red_state = red_state_layer(flattened_state_input)
    
    ## go through split
        ## trivial
trivial_red_state_layer = Dense(sillywalks.params.DIM_TRIVIAL_CONTROL, trainable=False, kernel_initializer='zeros')
trivial_red_state = trivial_red_state_layer(red_state)
        ## residual
residual_red_state_layer = Dense(sillywalks.params.DIM_RED_STATE-sillywalks.params.DIM_TRIVIAL_CONTROL, trainable=False, kernel_initializer='zeros')
residual_red_state = residual_red_state_layer(red_state)
    
    ## part acting on trivial red_state
trivial_control_layer = Dense(sillywalks.params.DIM_TRIVIAL_CONTROL, trainable=False, kernel_initializer='zeros')
trivial_red_action = trivial_control_layer(trivial_red_state)
    
    ## part acting on residual state
residual_red_state_1 = Dense(32)(residual_red_state)
residual_red_state_2 = Activation('relu')(residual_red_state_1)
residual_red_state_3 = Dense(32)(residual_red_state_2)
residual_red_state_4 = Activation('relu')(residual_red_state_3)
residual_last_layer = Dense(sillywalks.params.DIM_RED_ACTION-sillywalks.params.DIM_TRIVIAL_CONTROL, trainable=True, kernel_initializer='zeros')
residual_red_action = residual_last_layer(residual_red_state_4)

red_action = concatenate([trivial_red_action, residual_red_action])
inverse_reduce_action_layer = Dense(sillywalks.params.DIM_ACTION, trainable=False, kernel_initializer='zeros')
action_output = inverse_reduce_action_layer(red_action)

actor = Model(inputs=[state_input,], outputs=action_output)
print 'actor summary'
print(actor.summary())

    ## set actor weights
red_state_layer.set_weights([sillywalks.params.REDUCE_STATE_W, sillywalks.params.REDUCE_STATE_B])
print 'red_state_layer initial weights ', red_state_layer.get_weights()

trivial_red_state_layer.set_weights([sillywalks.params.TRIVIAL_RED_STATE_W, sillywalks.params.TRIVIAL_RED_STATE_B])
print 'trivial_red_state_layer initial weights ', trivial_red_state_layer.get_weights()
residual_red_state_layer.set_weights([sillywalks.params.RESIDUAL_RED_STATE_W, sillywalks.params.RESIDUAL_RED_STATE_B])
print 'residual_red_state_layer initial weights ', residual_red_state_layer.get_weights()

trivial_control_layer.set_weights([sillywalks.params.TRIVIAL_CONTROL_W, sillywalks.params.TRIVIAL_CONTROL_B])
print 'trivial_control_layer initial weights ', trivial_control_layer.get_weights()
#residual_control_layer.set_weights([sillywalks.params.RESIDUAL_CONTROL_W, sillywalks.params.RESIDUAL_CONTROL_B])
#print 'residual_control_layer initial weights ', residual_control_layer.get_weights()

inverse_reduce_action_layer.set_weights([sillywalks.params.INVERSE_REDUCE_ACTION_W, sillywalks.params.INVERSE_REDUCE_ACTION_B])
print 'inverse_reduce_action_layer initial weights ', inverse_reduce_action_layer.get_weights()



##########################################################################################################
######################### critic: Q function, (state, action) to real ####################################

action_input = Input(shape=(sillywalks.params.DIM_ACTION,), name='action_input')
observation_input = Input(shape=(1,) + (sillywalks.params.DIM_STATE,), name='observation_input')
flattened_observation = Flatten()(observation_input)

red_action_layer = Dense(sillywalks.params.DIM_RED_ACTION)
red_action = red_action_layer(action_input)
red_state_layer = Dense(sillywalks.params.DIM_RED_STATE)
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

weights = red_state_layer.set_weights([sillywalks.params.REDUCE_STATE_W, sillywalks.params.REDUCE_STATE_B])
print 'reduce_state_layer ', red_state_layer.get_weights()
weights = red_action_layer.set_weights([sillywalks.params.REDUCE_ACTION_W, sillywalks.params.REDUCE_ACTION_B])
print 'get reduce_action_layer ', red_action_layer.get_weights()


##########################################################################################################
######################### agent ####################################

# Set up the agent for training

memory = SequentialMemory(limit=100000, window_length=1)
random_process = OrnsteinUhlenbeckProcess(theta=.015, mu=0., sigma=.15, size=sillywalks.params.DIM_ACTION)
ddpg_agent = DDPGAgent(nb_actions=sillywalks.params.DIM_ACTION, actor=actor, critic=critic, critic_action_input=action_input,
                  memory=memory, nb_steps_warmup_critic=100, nb_steps_warmup_actor=100,
                  random_process=random_process, gamma=.99, target_model_update=1e-3,
                  delta_clip=1.)
# agent = ContinuousDQNAgent(nb_actions=env.noutput, V_model=V_model, L_model=L_model, mu_model=mu_model,
#                            memory=memory, nb_steps_warmup=1000, random_process=random_process,
#                            gamma=.99, target_model_update=0.1)
ddpg_agent.compile(Adam(lr=.001, clipnorm=1.), metrics=['mae'])


