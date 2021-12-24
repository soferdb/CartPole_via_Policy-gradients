# CartPole_via_Policy-gradients
Solving CartPole-v1 environment using policy-gradient methods.


This repository contains the implementation of the following policy-gradient methods using tensorflow 1.4:
1.  Vanilla REINFORCE.
2.  REINFORCE with Advantage-function. (with Value-network trained using Target-Network & Replay-Buffer)
3.  Actor-Critic (A2C) using n-steps bootstrapping.

To run each configuration read the following instructions.

For Vanilla REINFORCE run the following command -
    python policy_gradients.py
    
For REINFORCE using Advantage trained **with** Replay-Buffer & Target-Network run the following command -
    python policy_gradients.py --N 0
    
For REINFORCE using Advantage trained **without** Replay-Buffer & Target-Network run the following command -
    python policy_gradients.py --N 501
    
For A2C using **n**-steps run the following command -
    python policy_gradients.py --N <n>

Optional commmand-line arguments:
1.  --exp <exp_name>,       to set experiment name, and save results under this name.
2.  --n_episodes <integer>, to set the number of max episodes
3.  --gpu <x>,              to set CUDA_VISIBLE_DEVICES environment variable, and use gpu indexed with x.

Our goal is to achieve average-score over the past 100 episodes of 475.
The algortihm stops when the goal is reached.
  
  
Results of the different algorithms can be seen in the following graphs.
Details, explanations are provided within the attached PDF file.
  
 ![avg_scores_1](https://user-images.githubusercontent.com/49614331/147349078-eee18099-bec2-437f-a7be-602335894c45.PNG)
![avg_scores](https://user-images.githubusercontent.com/49614331/147349083-48825d8d-2e29-4cee-a03d-bf90f51cfd65.png)
  ![AC_avg_scores](https://user-images.githubusercontent.com/49614331/147349082-cac850fc-ab55-426c-a7a8-9852d58ee3aa.png)


  
