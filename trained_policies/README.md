# Policy Training

A large number n>100 of PPO~\cite{schulman_proximal_2017} policies were trained in each of the MinAtar environments 
using a close adaption to the training algorithm presented in the example folder on the [Pgx GitHub
page](https://github.com/sotetsuk/pgx/tree/main/examples/minatar-ppo), i.e. `train_ppo_agents.py` in this repository. 
The trained policies were trained with the same random seeding of the environments and action choices for 200M environment steps, with \num{6.25}M parameter updates using Adam with a learning rate of \num{0.0003}. 
The policy architectures were identical for all policies in all environments: a 32 2x2 ReLU-activated convolutional filters followed by 2x2 non-overlapping average pooling and a ReLU-activated linear layer of width 64, before the actor and critic parts diverge into two separate structures of two following ReLU-activated 64-width linear layers. The final layer of the critic is a width 1 linear layer, while the final layer of the actor is a layer of width corresponding to the number of actions available to the policy in the specific environment. The softmax function transforms the outputs of the actor into a probability distribution, and during training and evaluation, it draws the action choices stochastically from the probability distribution over the actions. 
During training, this seed was also fixed. More details on the architecture structure and learning scheme are available in the Pgx Github folder.  
