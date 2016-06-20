# DQN in Keras + TensorFlow + OpenAI Gym
This is an implementation of DQN in Keras + TensorFlow + OpenAI Gym.  

## Requirements
- gym
- scikit-image
- keras
- tensorflow

## Results
This is the result of training of DQN for about 20 hours(10000 episodes/4.35 millions frame) on AWS EC2 g2.2xlarge instance.  
![result](assets/result.gif)
<br>
Statistics of average loss, average max q values, total reward and duration / episode.  
![result](assets/result.png)

## Usage
#### Training
For DQN, run:

```
python dqn.py
```

For Double DQN, run:

```
python ddqn.py
```

#### Visualizing learning with TensorBoard
Run the following:

```
tensorboard --logdir=summary/[filename]
```

## Using GPU
I made an AMI for this experiment. All of requirements and CUDA and cuDNN are pre-installed in the AMI.  
The AMI name is `DQN-AMI` and ID is `ami-c4a969a9`. Feel free to use it.  

## References
- [Human-level control through deep reinforcement learning](http://www.nature.com/nature/journal/v518/n7540/full/nature14236.html)
- [Deep Reinforcement Learning with Double Q-learning](http://arxiv.org/abs/1509.06461)
- [songrotek/DQN-Atari-Tensorflow](https://github.com/songrotek/DQN-Atari-Tensorflow)
- [devsisters/DQN-tensorflow](https://github.com/devsisters/DQN-tensorflow)
- [coreylynch/async-rl](https://github.com/coreylynch/async-rl)
