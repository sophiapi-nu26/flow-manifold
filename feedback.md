Thanks — this looks strong overall, and I think the implementation is aligned with the intended experiment. In particular, I’m happy with the fixed-GMM / fixed-(U) policy, the use of oracle marginal velocity MSE as the main metric, the nested datasets, and the deduplication to 54 unique runs rather than 63. 

I’m comfortable approving this for a pilot phase.

Before the full sweep, I’d like two small changes / checks:

1. Please make the trajectory-language fully consistent everywhere. The report describes forward integration from Gaussian noise at (t=0.01) to (t=0.99), which is fine, but I want to avoid any conflicting “reverse ODE” wording in comments, docs, or figure text. 

2. Please pin evaluation randomness for the reported metrics — either by fixing an eval seed or by caching the evaluation tuples ((x_1, x_0, t)). Since the report notes that oracle MSE currently has small Monte Carlo variance from re-sampling evaluation points, I want the final paper numbers to be exactly reproducible. 

Launch conditions from my side:

* run one realistic full-epoch pilot locally,
* run one timing pilot for a large (d_x=512) config,
* then we’ll launch the full 54-run Slurm sweep if those look clean. 

Also, please do not make per-config changes like reducing epochs or widening the model unless we decide to do that uniformly across the whole sweep and document it clearly. 

Nice work on this.
