# Dirichlet-Process Models for Mars Spectroscopy

***The codebase is used for ongoing research. We open sourced the core algorithm that would benefit the most from parallelizing.***

**Staff Supervisors:** Prof. Dr. Cara Magnabosco, Dr. Adamos Valantinas

## Introduction

Over 17 years of data collection by the Mars Reconnaissance Orbiter has produced, among other products, two key datasets: the Multispectral Reduced Data Record (MRDR) and the VNIR Hyperspectral Reduced Data Record (VRDR) (Seelos et al., 2024). The MRDR dataset covers wavelengths from 0.36 to 3.92 µm, while the VRDR dataset spans the 0.4 to 1.02 µm range. Key minerals on Mars have spectral fingerprints in this region, making it crucial for understanding the Mars environment. We use the Jezero region as ground truth and want to extend our analysis globally. This is the recent landing site of NASA's Perseverance rover, which is looking for signs of ancient life and reconstructing the geologic history of the crater. These datasets, spanning multiple terabytes, enable researchers to perform data-driven spectral analysis to understand the geological properties of Mars. In our project, we implement data processing and clustering pipelines that process the entire planet-scale data. Currently, our sequential algorithm terminates in 14 days. Because the final output is sensitive to preprocessing choices and hyperparameter settings, a speedup of even one order of magnitude would let us iterate over many combinations rather than commit to one.

Nonparametric Bayesian clustering models are commonly used in this domain. Variants of the Hierarchical Dirichlet Process (HDP) have so far been applied either at small scale (Plebani et al., 2022), as a preprocessing step (Platt et al., 2026), or to facilitate downstream manual labelling (Dundar et al., 2019). In contrast, we leverage the full VRDR archive and use HDP as the final step of our clustering pipeline to label regions that have received little prior attention. We validate qualitatively by comparing the global output against well-studied regions such as Jezero.


## Hierarchical Dirichlet-Process Models

Conceptually, a Dirichlet Process (DP) is a distribution over distributions. Sampling from a DP yields a discrete distribution with infinitely many clusters, but in any finite dataset only a finite (and data-driven) number of those clusters are actually used. A concentration parameter governs how readily new clusters form as more data arrives. One of the main benefits over using standard $K$-means is that the algorithm is able to discover an appropriate number of clusters from the data.

The HDP is a natural extension of the DP that allows clustering with multiple data sets (Kulis & Jordan, 2012). We use HDP to implement a tractable online learning algorithm to be able to fit planet-scale data by streaming geographic tiles.

Mathematically, an HDP model can be summarised by:

<div align="center">

$$
\begin{aligned}
G_0 &\sim \mathrm{DP}(\gamma, H) \\
G_j &\sim \mathrm{DP}(\alpha, G_0) && \text{for } j = 1, \ldots, D \\
\phi_{ij} &\sim G_j && \text{for all } i, j \\
\mathbf{x}_{ij} &\sim \mathcal{N}(\phi_{ij}, \sigma I) && \text{for all } i, j
\end{aligned}
$$

</div>

where $G_0$ is the top-level Dirichlet process with concentration parameter $\gamma$ and base measure $H$; $G_j$ is the group-specific Dirichlet process for a planet tile $j$, with concentration parameter $\alpha$ and base measure $G_0$; $D$ is the number of tiles; $\phi_{ij}$ is the cluster parameter (mean) assigned to the $i$-th observation in tile $j$; and $\mathbf{x}_{ij}$ is the corresponding observation, modeled as Gaussian with mean $\phi_{ij}$ and isotropic covariance $\sigma I$.

### Implementation

Scaling the training of Bayesian nonparametric models, especially HDP, to large datasets is a well-known difficult problem. One option is to select a representative subset of the training data, referred to as a coreset (Bachem et al., 2015, Har-Peled & Mazumdar, 2018). Another option is to exploit parallelism (Dinari & Freifeld, 2022) and distribute the training (Balcan et al., 2020). 

A Hierarchical Dirichlet Process is fit by streaming geographic tiles through
two coupled DP-Means levels:

- A **local** DP-Means pass discovers tile-specific cluster centers from the
  flattened, NaN-masked feature matrix. The penalty `lambda_local` is set by a
  configurable percentile of pairwise sq-distances on a sample.
- A **global** merge step compares each newly discovered local center against
  the running set of global centers. If the nearest global center is farther
  than `lambda_global`, the local center SPAWNs a new global cluster; otherwise the
  global center is updated by a size-weighted average (MERGE).

```python
procedure HDP(tile, mode):
    # 1. Build valid-pixel mask, flatten to feature matrix X (N x D).

    if mode == INFERENCE:
        # 2. Squared distances X -> global centers G.
        # 3. argmin per pixel.
        # 4. Scatter back to (H, W) using the mask.
        return labels

    # -- Training branch --
    # 5. CORE_DP_MEANS(X, lambda_local) -> local centers C, per-pixel labels.
    # 6. Per-cluster sizes s_k = #{ pixel : label = k }.

    # 7. For each local cluster k (sequential — G mutates in-place):
    for k in local_clusters:
        if G is empty:
            # SEED — first tile only.
        else:
            d_star, j_star = nearest(C[k], G)
            if d_star > lambda_global:
                # SPAWN — append C[k] as a new global center.
            else:
                # MERGE — weighted update of G[j_star] using sizes (n[j_star], s_k).
    return updated global state


procedure CORE_DP_MEANS(X, lambda):  # local only, runs the inner DP-Means loop
    C <- {mean(X)}
    while not converged:
        # a. Per-pixel min-distance to C (compute-bound, GEMM-shaped).
        # b. Global argmax over those minimums (block reduction).
        # c. If max > lambda: promote that pixel to a new center (SERIAL hazard).
        # d. Reassign every pixel to its nearest center.
        # e. Recompute every center as the mean of its assigned pixels (segmented reduce).
    return C, labels
```

## Parallelization targets

The inner kernel structure breaks into three regimes with different GPU
characteristics:

- **Compute-bound (GEMM-shaped)**
- **Reduction-pattern:** per-pixel `argmin` (step a) and the global `argmax`
  (step b) are warp / block reductions.
- **Bandwidth-bound, irregular:** centroid recomputation (step e) and the final
  scatter back to the (H, W) tile are segmented reductions and a sparse write.

Four bottlenecks identified for the hackathon:

1. **Outer loop in `CORE_DP_MEANS` is serially iterative** — each iteration may
   promote a new center that affects later assignments. Candidate techniques:
   batched promotion, mini-batch variant, or algorithmic restructuring.
2. **Host–device sync from dynamic K** — pre-allocate `K_max` slots with a
   device-side counter to avoid copy-back per promotion.
3. **MERGE/SPAWN loop (step 7)** — the per-tile local cluster count is
   `O(10-100)`, tiny relative to pixel work. Decide between keeping it on the host or fusing it on-device.
4. **Centroid recomputation label skew** — a few clusters typically contain the majority of pixels. Sort-by-label + segmented reduce is a candidate; benchmark against a native atomic-based implementation.


### Serial Implementation Results from Our Upstream Project

Note that in our case the HDP algorithm is part of a larger pipeline. Our serial pipeline runtime for 1 VRDR tile obtains the following times:

| Phase | Absolute Time | % of Total Time |
| :--- | :--- | :--- |
| **Total Algorithm Time** | 1637.72s | 100% |
| **HDP Model** | 1590.07s | 97.1% |
| **Preprocessing** | 26.94s | 1.6% |
| **Visualization** | 20.61s | 1.3% |

*experiment setup: running on a personal laptop with an Intel(R) Core(TM) Ultra 7 155U*

The execution of the HDP model is a severe bottleneck in the processing pipeline. With around 2 thousand tiles in the full archive, this extrapolates to roughly 14 days end-to-end. GPU acceleration could drastically accelerate the overall execution, enabling new experiments and the ability to iterate over different preprocessing techniques.

We conclude that it is valuable for scientific application to accelerate the implementation of the HDP algorithm. To the best of our knowledge, no GPU implementation that mirrors our serial one exist.

## Layout

```
src/
  algorithm.py     # HierarchicalDPMeans: partial_fit / predict / reset
  synthetic.py     # Mars-like tile generator (mixture-of-Gaussians + NaN holes)
  benchmark.py     # Serial CPU baseline benchmark
tests/
  test_algorithm.py
requirements.txt
```

## Quickstart

```bash
pip install -r requirements.txt

# Serial baseline benchmark (~2 min)
python src/benchmark.py --tiles 4 --height 384 --width 384

# Smoke tests
python tests/test_algorithm.py
```

`HierarchicalDPMeans` exposes an API:

```python
from algorithm import HierarchicalDPMeans
from synthetic import TileSpec, generate_stream

model = HierarchicalDPMeans(local_percentile=50.0, global_percentile=30.0)

# Stream tiles: each tile is np.ndarray of shape (H, W, D), NaN = invalid.
for tile in generate_stream(n_tiles=8, spec=TileSpec()):
    labels = model.partial_fit(tile)   # (H, W) int32, -1 on invalid pixels

# After the stream, predict() reuses the converged global centers.
final_labels = model.predict(some_held_out_tile)
```

## Synthetic data

The `synthetic` module produces tiles drawn from a shared mixture of Gaussians in feature space, with a configurable NaN fraction simulating CRISM invalid pixels. The shared mixture is what lets the streaming model converge to a coherent global cluster set across tiles. Per-tile cluster weights are sampled from a Dirichlet so different tiles have different dominant clusters. This exercises the SPAWN/MERGE logic the way real-world geographic heterogeneity does.

Real CRISM tiles after PCA-14 are `~600 × 600 × 14` with ~10–30% invalid
pixels; the defaults in `benchmark.py` are sized similarly. Bump `--tiles` and
the spatial dimensions to scale the workload.

## Hackathon Targets

- **Primary:** writing a CUDA version for the streaming HDP algorithm
- **Secondary:** obtain an end-to-end pipeline speedup on a single GPU.
- **Stretch:** multi-GPU run that brings the full-archive pass under 24 hours, making hyperparameter sweeps tractable


### Team and prior experience.

1. Andrei Girjoaba (agirjoaba@ethz.ch): The main contributor. Developed the algorithm as a research project in the Department of Earth and Planetary Sciences. Computer Science MSc student at ETH, major in data mangement systems, specializing in heterogeneous computer architectures. Took relevant courses such as Advanced Systems Lab and Design of Parallel and High-Performance Computing (grade: 6/6).
2. Josh Anderegg (janderegg@student.ethz.ch): Experienced with the development of remote sensor pipelines through a project at the Swiss Federal Institute of Aquatic Science and Technology concerned with using satellite images for mining detection. Computer Science MSc student at ETH. Specialized in systems. Took relevant courses such as dvanced Systems Lab and Design of Parallel and High-Performance Computing (grade: 5.75/6).
3. Elena Mihalache (e.mihalache@student.tudelft.nl): During the MSc thesis, gained experience with performance analysis and energy consumption for CPUs. Wants to learn more about parallel hardware. Specialized in software testing and will contribute by guaranteeing the correctness of the parallel implementation. Computer Science Master student at TU Delft, majoring in data management systems and software engineering.
4. Vladimir Sachkov (vladimirsachkov2003@gmail.com): Experience with remote sensing data and classical computer vision. Deployed machine learning pipelines into production for large incoming traffic on consumer grade GPUs and algorithmical optimisations at Clear Timber Analytics in the Netherlands as lead software engineer. 


## References

Seelos, F. P., Seelos, K. D., Murchie, S. L., Novak, M. A. M., Hash, C. D., Morgan, M. F., Arvidson, R. E., Aiello, J., Bibring, J.-P., Bishop, J. L., Boldt, J. D., Boyd, A. R., Buczkowski, D. L., Chen, P. Y., Clancy, R. T., Ehlmann, B. L., Frizzell, K., Hancock, K. M., Hayes, J. R., … Wolff, M. J. (2024). The CRISM investigation in Mars orbit: Overview, history, and delivered data products. Icarus, 419, 115612. https://doi.org/10.1016/j.icarus.2023.115612

Dundar, M., Ehlmann, B., & Leask, E. (2019). RARE PHASE DETECTIONS IN CRISM DATA AT PIXEL-SCALE BY MACHINE LEARNING GENERATE NEW DISCOVERIES ABOUT GEOLOGY AT MARS ROVER LANDING AREAS: JEZERO AND NE. https://www.semanticscholar.org/paper/RARE-PHASE-DETECTIONS-IN-CRISM-DATA-AT-PIXEL-SCALE-Dundar-Ehlmann/4a990146ca1abca2413ed11b0e49030d67c1ee5d

Plebani, E., Ehlmann, B. L., Leask, E. K., Fox, V. K., & Dundar, M. M. (2022). A machine learning toolkit for CRISM image analysis. Icarus, 376, 114849. https://doi.org/10.1016/j.icarus.2021.114849

Platt, R., Arcucci, R., & John, C. M. (2026). Mineralogical Classification of CRISM Hyperspectral Data Under Uncertainty With Hybrid Neural Networks. Journal of Geophysical Research: Planets, 131(3), e2025JE009473. https://doi.org/10.1029/2025JE009473

Kulis, B., & Jordan, M. I. (2012). Revisiting k-means: New Algorithms via Bayesian Nonparametrics (arXiv:1111.0352). arXiv. https://doi.org/10.48550/arXiv.1111.0352

Bachem, O., Lucic, M., & Krause, A. (2015). Coresets for Nonparametric Estimation—The Case of DP-Means. Proceedings of the 32nd International Conference on Machine Learning, 209–217. https://proceedings.mlr.press/v37/bachem15.html

Har-Peled, S., & Mazumdar, S. (2018). Coresets for $k$-Means and $k$-Median Clustering and their Applications (arXiv:1810.12826). arXiv. https://doi.org/10.48550/arXiv.1810.12826

On Coresets for k-Median and k-Means Clustering in Metric and Euclidean Spaces and Their Applications. (n.d.). SIAM Journal on Computing. Retrieved April 27, 2026, from https://epubs.siam.org/doi/10.1137/070699007

Dinari, O., & Freifeld, O. (2022). Revisiting DP-Means: Fast scalable algorithms via parallelism and delayed cluster creation. Proceedings of the Thirty-Eighth Conference on Uncertainty in Artificial Intelligence, 579–588. https://proceedings.mlr.press/v180/dinari22b.html

Balcan, M. F., Ehrlich, S., & Liang, Y. (2020). Distributed k-Means and k-Median Clustering on General Topologies (arXiv:1306.0604). arXiv. https://doi.org/10.48550/arXiv.1306.0604


