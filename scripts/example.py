# keras-rl train-test setup
import opensim as osim
from osim.http.client import Client
import numpy as np
import argparse

import sillywalks.controllers
import sillywalks.env


############################################################################################
## setting params

DIFFICULTY = 0

# Command line parameters
parser = argparse.ArgumentParser(description='Train or test neural net motor controller')
parser.add_argument('--train', dest='train', action='store_true', default=True)
parser.add_argument('--test', dest='train', action='store_false', default=True)
parser.add_argument('--steps', dest='steps', action='store', default=10000, type=int)
parser.add_argument('--visualize', dest='visualize', action='store_true', default=False)
parser.add_argument('--model', dest='model', action='store', default="/models/example.h5f")
parser.add_argument('--loadweights', dest='loadweights', action='store_true', default=False)
parser.add_argument('--token', dest='token', action='store', required=False)
args = parser.parse_args()
# Total number of steps in training
nallsteps = args.steps



############################################################################################
# loading environment
env = sillywalks.env.RunEnv(args.visualize)
state = env.reset(difficulty = DIFFICULTY, seed = None) 
print 'initial state: ', state

## loading agent from agent module
agent = sillywalks.agents.ddpg_agent

if args.loadweights:
    print '#############################################################################################'
    print 'loading weights from ', args.model
    print '#############################################################################################'
    agent.load_weights(args.model)




############################################################################################################
## fit/train/test

if args.train:
    agent.fit(env, nb_steps=nallsteps, visualize=False, verbose=1, nb_max_episode_steps=env.timestep_limit, log_interval=5000)
    # After training is done, we save the final weights. ##TODO, this sucks
    agent.save_weights(args.model, overwrite=True)

# If TEST and TOKEN, submit to crowdAI
if not args.train and args.token:
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
    # Finally, evaluate our algorithm for 1 episode.
    agent.test(env, nb_episodes=1, visualize=True, nb_max_episode_steps=500)


