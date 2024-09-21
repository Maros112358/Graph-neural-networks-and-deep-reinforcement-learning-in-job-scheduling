# Literature

## Recommended literature

### [Gated-Attention Model with Reinforcement Learning for Solving DynamicJob Shop Scheduling Problem](https://onlinelibrary.wiley.com/doi/epdf/10.1002/tee.23788)

- did not find the source code for the model

### [Learning to schedule job-shop problems: Representation and policy learning using graph neural network and reinforcement learning](https://arxiv.org/abs/2106.01086)

- did not find the source code for the model

### [Large-Scale Dynamic Scheduling for Flexible Job-Shop With Random Arrivals of New Jobs by Hierarchical Reinforcement Learning](https://ieeexplore.ieee.org/document/10114974)

- did not find the source code for the model

### [Learning to Dispatch for Job Shop Scheduling via Deep Reinforcement Learning](https://browse.arxiv.org/pdf/2010.12367.pdf)

- official implementation: [L2D](models/L2D/)
- formulates dispatching as Markov Decision Process with reward minimizing the makespan
- the agent is designed as Graph Isomorphism Network (GIN), which is a recent GNN variant

### [Flexible Job Shop Scheduling via Graph Neural Network and Deep Reinforcement Learning](https://ieeexplore.ieee.org/document/9826438)

- official implementation: [fjsp-drl](models/fjsp-drl/)
- works with Flexible Job-shop Scheduling Problem (FJSP) which is a more general case than JSSP
- formulates dispatching as Markov Decision Process with reward minimizing the makespan
- the formulation of FJSP ased on a novel heterogeneous graph representation of scheduling states
- the agent uses a heterogeneous-graph-neural-network (HGNN)

### [Deep Reinforcement Learning Based on Graph Neural Networks for Job-shop Scheduling](https://ieeexplore.ieee.org/abstract/document/10226873)

- github: https://github.com/Jerry-Github-Cloud/IEEE-ICCE-RL-JSP

## Other literature

### [Job-Shop Scheduling](https://www.researchgate.net/publication/2244529_Job-Shop_Scheduling)

### [Combining Reinforcement Learning Algorithms with Graph Neural Networks to Solve Dynamic Job Shop Scheduling Problems](https://www.researchgate.net/publication/370959257_Combining_Reinforcement_Learning_Algorithms_with_Graph_Neural_Networks_to_Solve_Dynamic_Job_Shop_Scheduling_Problems)

- no code found

### [A Reinforcement Learning Environment For Job-Shop Scheduling](https://browse.arxiv.org/pdf/2104.03760.pdf)

- #TODO: Read

## Github repo

## [Master's Thesis - Graph Neural Networks for Compact Representation for Job Shop Scheduling Problems: A Comparative Benchmark](https://github.com/MattJud/gnn_jssp)

## [Masters Thesis - Devloping the scheduler for the standard JSSP using GNN based RL.](https://github.com/sachin301195/Thesis/tree/main)

## [Implementations of combinatorial optimization ML papers](https://github.com/valeman/awesome-ml4co/blob/841a4f24c893bf45e83147414548763b22e54685/data/papers.csv#L166)

## Interesting sources 

### [Masters Thesis - Devloping the scheduler for the standard JSSP using GNN based RL.](https://github.com/sachin301195/Thesis/tree/main)

### [RL-Scheduling](https://github.com/hliangzhao/RL-Scheduling)

- someone tried to implement the algorithm Decima, published in SIGCOMM '19 (https://web.mit.edu/decima/). 
- repository almost 3 years without activity

### [DRL-for-Job-Shop-Scheduling](https://github.com/hexiao5886/DRL-for-Job-Shop-Scheduling/tree/master)

- repository doing similar research as this thesis, may be useful
- last activity during May
