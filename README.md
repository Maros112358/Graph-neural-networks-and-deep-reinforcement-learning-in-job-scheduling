# [diplomka68.html](https://www.cs.cas.cz/~martin/diplomka68.html)

## Models

### [Wheatley](models/Wheatley/)

- [SHOWCASE](models/Wheatley/repo/Showcase.ipynb)
- trained 15x15 Wheatley model, showcase for small benchmark instances 
- dynamic JSSP implemented by adding "virtual machines" at the start of the jobs


### [fjsp-drl](models/fjsp-drl/)

- [SHOWCASE](models/fjsp-drl/repo/Showcase%20fjsp-drl.ipynb)
- model runs for all available benchmarks
- no dynamic version

### [L2D](models/L2D/)

- [SHOWCASE](models/L2D/Showcase.ipynb)
- model runs for all available benchmarks
- dynamic version runs by implementing custom algorithm
- [STATIC EXPERIMENT DATA](data/experiment_static_jssp_l2d.csv)

### [IEEE-ICCE-RL-JSP](models/IEEE-ICCE-RL-JSP/)

- [SHOWCASE](models/IEEE-ICCE-RL-JSP/repo/ieee_icce_rl_jsp.ipynb)
- I trained 5 models running on all available benchmarks
- working dynamic model implemented by using already available code, just needed to pass the starting times to the underlying mechanism already in the code by the author
- [STATIC EXPERIMENT DATA](data/experiment_static_jssp_ieee_icce_rl_jsp.csv)

### [End-to-end-DRL-for-FJSP](models/End-to-end-DRL-for-FJSP/)

- [SHOWCASE](models/End-to-end-DRL-for-FJSP/repo/FJSP_RealWorld/Showcase.ipynb)
- model runs for all available benchmarks
- no dynamic version


## Benchmarks

### [JSSP](benchmarks/jssp/)

Description of JSSP instances is available at http://jobshop.jjvh.nl/explanation.php

- [Adams, Balas and Zawacks](benchmarks/jssp/abz_instances)

- [Demirkol, Mehta, and Uzsoy](benchmarks/jssp/dmu_instances/)

- [Fisher and Thompson](benchmarks/jssp/ft_instances/)

- [Lawrence](benchmarks/jssp/la_instances/)

- [Applegate and Cook](benchmarks/jssp/orb_instances/)

- [Storer, Wu and Vaccari](benchmarks/jssp/swv_instances/)

- [Taillard](benchmarks/jssp/ta_instances/)

- [Yamada and Nakano](benchmarks/jssp/yn_instances/)

### [FJSP](benchmarks/fjsp/)

Description of FJSP instances is available at [DataSetExplanation.txt](benchmarks/fjsp/DataSetExplanation.txt)

- [Behnke and Geiger](benchmarks/fjsp/0_BehnkeGeiger/)

- [Instances for the general FJSSP by Brandimarte](benchmarks/fjsp/1_Brandimarte/)

- [MPM-JSSP-instances by Hurink et al.](benchmarks/fjsp/2_Hurink/)

- [Multiprocessor-JSSP/FMS-instances by Dauzère-Pérès and Paulli](benchmarks/fjsp/3_DPpaulli/)

- [FJSSP-instances by Chambers and Barnes](benchmarks/fjsp/4_ChambersBarnes/)

- [Instances for the general FJSSP with total flexibility by Kacem et al.](benchmarks/fjsp/5_Kacem/)

- [FJSSP-instances for Mathematical Programming by Fattahi et al.](benchmarks/fjsp/6_Fattahi/)

## Interesting sources 

### [Masters Thesis - Devloping the scheduler for the standard JSSP using GNN based RL.](https://github.com/sachin301195/Thesis/tree/main)

### [RL-Scheduling](https://github.com/hliangzhao/RL-Scheduling)

- someone tried to implement the algorithm Decima, published in SIGCOMM '19 (https://web.mit.edu/decima/). 
- repository almost 3 years without activity

### [DRL-for-Job-Shop-Scheduling](https://github.com/hexiao5886/DRL-for-Job-Shop-Scheduling/tree/master)

- repository doing similar research as this thesis, may be useful
- last activity during May
