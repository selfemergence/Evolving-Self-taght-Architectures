# -*- coding: utf-8 -*-

import numpy as np
import gym
from keras import models
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import RMSprop

# hyperparameters
H = 200 # number of hidden layer neurons
batch_size = 10 # every how many episodes to do a param update?
learning_rate = 1e-4
gamma = 0.99 # discount factor for reward
decay_rate = 0.99 # decay factor for RMSProp leaky sum of grad^2
resume = False # resume from previous checkpoint?
render = False

# model initialization
D = 80 * 80 # input dimensionality: 80x80 grid

model = models.Sequential()
model.add(Dense(D, input_shape=(D,)))
model.add(Dense(200, activation = 'relu'))
model.add(Dense(1, activation='sigmoid'))
rms = RMSprop(lr=learning_rate)

model.compile(loss='binary_crossentropy', optimizer=rms)

if resume:
    model.load_weights("weights.h5")
    
def prepro(I):
  """ prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
  I = I[35:195] # crop
  I = I[::2,::2,0] # downsample by factor of 2
  I[I == 144] = 0 # erase background (background type 1)
  I[I == 109] = 0 # erase background (background type 2)
  I[I != 0] = 1 # everything else (paddles, ball) just set to 1
  return I.astype(np.float).ravel()

def discount_rewards(r):
  """ take 1D float array of rewards and compute discounted reward """
  discounted_r = np.zeros_like(r)
  running_add = 0
  for t in reversed(range(0, r.size)):
    if r[t] != 0: running_add = 0 # reset the sum, since this was a game boundary (pong specific!)
    running_add = running_add * gamma + r[t]
    discounted_r[t] = running_add
  return discounted_r

env = gym.make("Pong-v0")
observation = env.reset()
prev_x = None # used in computing the difference frame
xs,dlogps,drs = [],[],[]
running_reward = None
reward_sum = 0
episode_number = 0


while True:
  if render: 
      env.render()

  # preprocess the observation, set input to network to be difference image
  cur_x = prepro(observation)
  x = cur_x - prev_x if prev_x is not None else np.zeros(D)
  prev_x = cur_x
  
   # forward the policy network and sample an action from the returned probability
  aprob = model.predict(x.reshape([1,x.shape[0]]), batch_size=1)
  action = 2 if np.random.uniform() < aprob else 3 # roll the dice!
  
  # record various intermediates (needed later for backprop)
  xs.append(x) # observation
  y = 1 if action == 2 else 0 # a "fake label"
  dlogps.append(y - aprob)
  
  # step the environment and get new measurements
  observation, reward, done, info = env.step(action)
  reward_sum += reward

  drs.append(reward) 
  
  if done: # an episode finished
    episode_number += 1

    # stack together all inputs, hidden states, action gradients, and rewards for this episode
    epx = np.vstack(xs)
    epdlogp = np.vstack(dlogps)
    epr = np.vstack(drs)
    xs,dlogps,drs = [],[],[] # reset array memory

    # compute the discounted reward backwards through time
    discounted_epr = discount_rewards(epr)

    # standardize the rewards to be unit normal (helps control the gradient estimator variance)
    discounted_epr -= np.mean(discounted_epr)
    discounted_epr /= np.std(discounted_epr)

    epdlogp *= discounted_epr # modulate the gradient with advantage (PG magic happens right here.)
    grad = model.train_on_batch(epx, epdlogp)

    # boring book-keeping
    running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
    print('resetting env. episode reward total was %f. running mean: %f' % (reward_sum, running_reward))
    
    # if episode_number % 100 == 0: pickle.dump(model, open('save.p', 'wb'))
    if episode_number % 100 == 0: model.save_weights("weights.h5")
    reward_sum = 0
    observation = env.reset() # reset env
    prev_x = None
  if reward != 0: # Pong has either +1 or -1 reward exactly when game ends.
    print(('ep %d: game finished, reward: %f' % (episode_number, reward)) + ('' if reward == -1 else ' !!!!!!!!'))