""" 
-----------------------------------
For train.py                      #
-----------------------------------
"""
WARM_POPULATE = 0
TRAINER_MAX_EPOCHS = 150000
VAL_CHECK_INTERVAL = 100
TRAIN = False

""" 
-----------------------------------
For agent.py                      #
-----------------------------------
"""
OU_NOISE_STD = 0.8

""" 
-----------------------------------
For buffers.py                    #
-----------------------------------
"""
ALPHA = 0
BETA = 0.1

""" 
-----------------------------------
For ddpg.py                       #
-----------------------------------
"""
ENV = "Pendulum-v1"  # "LunarLanderContinuous-v2"
USE_PRIORITIZED_BUFFER = False
BATCH_SIZE = 64
ACTOR_LR = 1e-4
CRITIC_LR = 5e-4
GAMMA = 0.99
EPISODE_LENGTH = 200
TRAIN_EPISODES = 300
SYNC_RATE = 1
TAU = 5e-3
