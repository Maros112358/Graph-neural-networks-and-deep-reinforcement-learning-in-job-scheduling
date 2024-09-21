# Graph Neural Networks and Deep Reinforcement Learning in Job Scheduling

- diploma thesis download: https://dspace.cuni.cz/handle/20.500.11956/190558
- diploma thesis assignment: https://www.cs.cas.cz/~martin/diplomka68.html


## Repository structure

```
root/
├── benchmaks/        # Benchmarks used for testing the models
│   ├── jssp/         # Benchmarks for Job-Shop Scheduling (JSSP)
│   └── fjsp/         # Benchmarks for Flexible Job-Shop Scheduling (FJSP)
├── data/             # Data produced by testing the models on the benchmarks
│   ├── scripts/      # Scripts used for analysing the data (very messy, I do not advise reusing it)
│   └── experiment_{dynamic|static}_{jssp|fjsp}_<model>.csv      # Experimental data from the testing of <model> on dynamic|static version of jssp|fjsp 
├── models/           # Tested models
│   ├── baseline/     # Priority Dispatching Rules (PDRs) used as a baselines for the experimental comparison
│       ├── jssp/     # PDRs for JSSP used from the model IEEE-ICCE-RL-JSP 
│       └── jssp/     # PDRs for FJSP used from the model End-to-end-DRL-for-FJSP
│   └── <model>/      # Each folder is named after the model in the section ## Compared models below
└── literature.md     # Relevant literature.md
```

## Compared Models

### [Wheatley](models/Wheatley/)

- [SHOWCASE](models/Wheatley/repo/Showcase.ipynb)
- trained 15x15 Wheatley model, showcase for small benchmark instances 
- dynamic JSSP implemented by adding "virtual machines" at the start of the jobs
- [STATIC JSSP DATA](data/experiment_static_jssp_wheatley.csv)
- [DYNAMIC JSSP DATA](data/experiment_dynamic_jssp_wheatley.csv)

### [fjsp-drl](models/fjsp-drl/)

- [SHOWCASE](models/fjsp-drl/repo/Showcase%20fjsp-drl.ipynb)
- model runs for all available benchmarks
- no dynamic version
- [STATIC JSSP DATA](data/experiment_static_jssp_fjsp_drl.csv)
- [STATIC FJSP DATA](data/experiment_static_fjsp_fjsp_drl.csv)

### [L2D](models/L2D/)

- [SHOWCASE](models/L2D/Showcase.ipynb)
- model runs for all available benchmarks
- dynamic version runs by implementing custom algorithm
- [STATIC JSSP DATA](data/experiment_static_jssp_l2d.csv)
- [DYNAMIC JSSP DATA](data/experiment_dynamic_jssp_l2d.csv)

### [IEEE-ICCE-RL-JSP](models/IEEE-ICCE-RL-JSP/)

- [SHOWCASE](models/IEEE-ICCE-RL-JSP/repo/ieee_icce_rl_jsp.ipynb)
- I trained 5 models running on all available benchmarks
- working dynamic model implemented by using already available code, just needed to pass the starting times to the underlying mechanism already in the code by the author
- [STATIC JSSP DATA](data/experiment_static_jssp_ieee_icce_rl_jsp.csv)
- [DYNAMIC JSSP DATA](data/experiment_dynamic_jssp_ieee_icce_rl_jsp.csv)

### [End-to-end-DRL-for-FJSP](models/End-to-end-DRL-for-FJSP/)

- [SHOWCASE](models/End-to-end-DRL-for-FJSP/repo/FJSP_RealWorld/Showcase.ipynb)
- model runs for all available benchmarks
- no dynamic version
- [STATIC JSSP DATA](data/experiment_static_jssp_end_to_end_drl_for_fjsp.csv)
- [STATIC FJSP DATA](data/experiment_static_fjsp_end_to_end_drl_for_fjsp.csv)

### [Baselines](models/baseline/)

- [JSSP BASELINE DATA](data/experiment_static_jssp_baseline.csv) [IEEE-ICCE-RL-JSP](models/IEEE-ICCE-RL-JSP/)
- [FJSP BASELiNE DATA](data/experiment_static_fjsp_baseline.csv) (code taken from [End-to-end-DRL-for-FJSP](models/End-to-end-DRL-for-FJSP/))

## Benchmarks

### JSSP

Description of JSSP instances is available at http://jobshop.jjvh.nl/explanation.php

- [Adams, Balas and Zawacks](benchmarks/jssp/abz_instances)

- [Demirkol, Mehta, and Uzsoy](benchmarks/jssp/dmu_instances/)

- [Fisher and Thompson](benchmarks/jssp/ft_instances/)

- [Lawrence](benchmarks/jssp/la_instances/)

- [Applegate and Cook](benchmarks/jssp/orb_instances/)

- [Storer, Wu and Vaccari](benchmarks/jssp/swv_instances/)

- [Taillard](benchmarks/jssp/ta_instances/)

- [Yamada and Nakano](benchmarks/jssp/yn_instances/)

### FJSP

Description of FJSP instances is available at [DataSetExplanation.txt](benchmarks/fjsp/DataSetExplanation.txt)

- [Behnke and Geiger](benchmarks/fjsp/0_BehnkeGeiger/)

- [Instances for the general FJSSP by Brandimarte](benchmarks/fjsp/1_Brandimarte/)

- [MPM-JSSP-instances by Hurink et al.](benchmarks/fjsp/2_Hurink/)

- [Multiprocessor-JSSP/FMS-instances by Dauzère-Pérès and Paulli](benchmarks/fjsp/3_DPpaulli/)

- [FJSSP-instances by Chambers and Barnes](benchmarks/fjsp/4_ChambersBarnes/)

- [Instances for the general FJSSP with total flexibility by Kacem et al.](benchmarks/fjsp/5_Kacem/)

- [FJSSP-instances for Mathematical Programming by Fattahi et al.](benchmarks/fjsp/6_Fattahi/)

## [Relevant literature](/literature.md)
