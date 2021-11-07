import argparse
import os
import json
import itertools
from unityagents import UnityEnvironment
from dqn_agent import Agent
import train
import torch

ALGORITHMS = {
    "double_dqn", 
    "prioritized_replay"
}

parser = argparse.ArgumentParser(description="Train an agent to solve the banana environment")
parser.add_argument('file', metavar='f', help="Path to a json configuration file or a saved network file.")

if __name__ == '__main__':
    args = parser.parse_args()

    if not os.path.exists(args.file):
        raise Exception('Argument {} does not exist'.format(args.file))

    env = UnityEnvironment(file_name="./Banana_Windows_x86_64/Banana.exe")
    # get the default brain
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]

    _,ext = os.path.splitext(args.file)
    if ext == ".pth":
        print("Running Agent from {}".format(args.file))
        env_info = env.reset(train_mode=False)[brain_name]
        network_info = torch.load(args.file)
        agent = Agent(state_size=len(env_info.vector_observations[0]), action_size=brain.vector_action_space_size, seed=0, config=network_info['config'])
        agent.qnetwork_local.load_state_dict(network_info['net'])
        train.run(env, agent, brain_name)
    else:
        # train a new network
        with open(args.file) as f:
            info = json.load(f)
            
            # make each of the items into a list if it isn't already, so that we can 
            # tune hyper parameters
            hyper_config = {}
            for k,v in info.items():
                # create list if it isn't a list.  all other args should be scalars
                hyper_config[k] = v if type(v) == list else [v]        

            for i,values in enumerate(itertools.product(*hyper_config.values())):
                config = dict(zip(hyper_config.keys(), values))
                # for each of the algorithms, create a key with the boolean flag if it's on
                for algo in ALGORITHMS:
                    config[algo] = False
                
                for algo in config['algorithms']:
                    config[algo] = True

                print('\n{} Training with {}'.format(i+1, config))

                # reset the environment to get its parameters
                env_info = env.reset(train_mode=True)[brain_name]
                # initialize an agent
                agent = Agent(state_size=len(env_info.vector_observations[0]), action_size=brain.vector_action_space_size, seed=0, config=config)

                # train the agent
                train.dqn(config, env, agent, brain_name, '{}_{}.pth'.format(config['base_name'], i))
    env.close()