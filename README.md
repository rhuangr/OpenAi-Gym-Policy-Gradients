## Project Description
The repository contains files for different policy gradient algorithms: vanilla policy gradient, REINFORCE and PPO (proximal policy approximation. T The `network.py` file contains the code for neural networks used in the policy gradient algorithms
## Installation
1.  Clone the repository
2.  Depending on which file you would like to run paste the one of the commands below while in the cloned directory :thumbsup: .
	- `python PPO.py`
	- `python REINFORCE.py`
	- `python vanilla_policy_gradient.py`
## Notes
Currently, the programs written will try to optimize the rewards obtained in the CartPole environment.
If you wish to specify a different environment, specify the environment name you wish to train like so `train(env_name = *env_name*)`
