# Final Feedback on Oracle-(U) Experiment Plan

## Overall Assessment

This version of the plan is ready to implement.

The remaining issues are no longer about the scientific validity of the experiment. Instead, they are mostly ordinary implementation details that can be handled during coding and code review.

At this point, the protocol is specific enough that two different implementers would likely produce essentially the same experiment. 

## What Is Now Fully Specified

The current plan now clearly fixes:

* the distinction between the original ambient experiment and the oracle-(U) latent-space ablation,
* the exact decomposition-theorem motivation,
* the difference between training targets and oracle evaluation targets,
* the projected conditional FM training target,
* the oracle tangent velocity evaluation target,
* the Panel A shared-data protocol,
* the cached evaluation format for Panel A,
* the Figure 2C latent-generation protocol,
* the sliced Wasserstein reproducibility settings,
* the PCA fitting rule,
* the fallback plan for unstable Panel B trends,
* and the sanity checks to run before the full sweep. 

## Why the Plan Is Ready

### 1. The Scientific Story Is Clear

The plan now tells a coherent empirical story:

* the original ambient experiment reflects the full practical problem, including subspace recovery,
* the oracle-(U) experiment isolates the tangent-learning problem predicted by the decomposition theorem,
* and the contrast between the two experiments is itself meaningful.

This is exactly the kind of controlled empirical validation that reviewers were asking for. 

### 2. The Training / Evaluation Distinction Is Correct

The plan now correctly separates:

* the noisy projected conditional FM targets used during training,
* from the noiseless oracle marginal velocity used during evaluation.

That distinction is important scientifically and is now clearly stated in the plan. 

### 3. Panel A Is Now a Proper Controlled Invariance Check

Panel A is now carefully designed to isolate the effect of (d_x):

* same latent GMM,
* same latent samples,
* same cached evaluation set,
* same training seeds,
* different (U) matrices only at the embedding step.

That makes the interpretation much cleaner:
once the model only sees latent coordinates, changing ambient dimension should not materially change the tangent estimation problem. 

### 4. Figure 2 Is Operationally Clear

Figure 2 now has a clear purpose and a reproducible protocol:

* Figure 2A shows learned vs oracle latent velocity,
* Figure 2B shows convergence of tangent oracle MSE during training,
* Figure 2C shows whether the learned latent dynamics generate a reasonable endpoint distribution.

This is much stronger than the earlier drafts and should produce figures that are easy to interpret in the paper. 

## Minor Notes for Implementation

These do not require another design revision, but are worth keeping in mind during coding.

### 1. Figure 2A Could Use Separate Coordinate Panels if Needed

The current plan suggests flattening all latent coordinates into one scatter and coloring by coordinate index.

That is fine.

However, if the scatter becomes visually cluttered, it would also be reasonable to use:

* one subplot per coordinate,
* or a small grid of coordinate-wise scatters.

That is purely a visualization choice and does not affect the experiment itself. 

### 2. Figure 2C May Need More ODE Steps if Generation Looks Noisy

Euler integration with 100 steps is reasonable.

However, if the generated latent endpoint distribution looks unstable or overly noisy, the first thing to try is increasing to 250–500 Euler steps before changing anything deeper about the model. 

### 3. The Cached Panel A File Should Probably Include Metadata

The cached evaluation file for Panel A should ideally include:

* (d_0),
* evaluation seed,
* GMM parameters,
* number of cached points,
* and possibly a short version string.

That will make the cached file easier to inspect and reuse later. 

### 4. Figure 1 Should Emphasize Per-Dimension MSE

The main Figure 1 plots should use per-dimension tangent oracle MSE.

Total tangent MSE can still be reported:

* in an appendix,
* in a secondary panel,
* or in a small inset.

It would also help to explicitly mention in the caption that total MSE naturally scales with (d_0). 

## Final Takeaway

The plan is now scientifically sound, operationally specific, and ready to implement.

At this point, the remaining work is coding, debugging, and running the sweep — not redesigning the experiment. 
