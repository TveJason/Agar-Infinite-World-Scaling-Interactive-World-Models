# Gilles–Dini research integration

An external cumulative monitor is a plausible experiment for long rollouts: measure change, accumulate evidence, and stop further generation at an explicit threshold. This implementation does not derive from or validate a Navier–Stokes regularity result. It is not the amplification/disagreement integral or a calibrated probabilistic risk bound.

The executable proxy computes the mean absolute difference in RGB channel means between the previous ending frame and the new ending frame, divided by 255. It adds this nonnegative value to a cumulative total. At or above the threshold, the completed chunk is saved and generation stops before the next chunk. The chunk is not rejected or rolled back. Non-finite frames cause an error.

This detects coarse appearance changes, misses semantic identity and geometry failures, and can stop valid camera movement or lighting changes. The threshold of 1.0 is uncalibrated; the monitor is disabled by default. It does not improve generation or restore consistency. NumPy is the only monitor dependency and is already required upstream. No separate Gilles–Dini package is needed.

Next: compare fixed scene/action/seed sets with and without monitoring; annotate true drift, false stops and misses; test independent visual embeddings; measure latency, memory and consistency at 81, 401 and 1,041 frames. LPIPS, visual encoders and optical flow are possible future dependencies, but are not wired in because viewpoint changes require careful calibration.
