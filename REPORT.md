# Clustering Flight Trajectories Using HDBSCAN:
# Identifying Operational Patterns and Anomaly Candidates

## Abstract

Aircraft trajectory data provide a detailed record of real-world flight
operations and can be used to identify recurring route structures as well as
flights that deviate from common operational patterns. This study develops and
evaluates an unsupervised trajectory-clustering framework for flights operating
from Los Angeles International Airport (LAX) to John F. Kennedy International
Airport (JFK). The dataset contains 1,354 complete trajectories from January
and July 2026.

Each complete flight is treated as one clustering sample. Following data
validation and temporal harmonization, trajectories are represented using 100
equally spaced points along cumulative horizontal arc length. Pairwise
trajectory dissimilarity is calculated using symmetric Hausdorff distance with
Haversine point-to-point distances. The resulting 1,354 × 1,354 distance matrix
is clustered using a manually implemented HDBSCAN algorithm.

The final configuration, using `min_samples = 15` and
`min_cluster_size = 30`, identifies four recurring route families containing
1,059 trajectories and 295 HDBSCAN noise trajectories. The manual
implementation produces an identical partition to the equivalent
scikit-learn HDBSCAN configuration.

Several sensitivity experiments indicate that the principal clustering
structure is highly robust. Increasing trajectory representation resolution
from 100 to 200 points produces an Adjusted Rand Index (ARI) of 0.989996,
while replacing Hausdorff distance with discrete Fréchet distance produces an
ARI of 0.993846. Among trajectories assigned to clusters by both sensitivity
configurations, the partition agreement is exact.

The HDBSCAN noise trajectories are interpreted as anomaly candidates rather
than automatically abnormal operations. Of the 295 primary noise trajectories,
239 remain noise across three neighboring HDBSCAN parameter configurations.
A preliminary post-clustering analysis also identifies an unusually strong
concentration of persistent and geometrically extreme trajectories on
28 July 2026. The cause of this event has not yet been established and will
require validation using higher-resolution ADS-B and independent meteorological
or operational information.

The results demonstrate that density-based clustering with trajectory-level
distance measures can provide interpretable and robust representations of
long-haul operational route structure while also identifying candidates for
subsequent event-level anomaly investigation.

---

## 1. Introduction

### 1.1 Background

Automatic Dependent Surveillance–Broadcast (ADS-B) systems provide
time-dependent aircraft state information that enables large-scale
investigation of real-world flight operations. The OpenSky Network has made
large quantities of ADS-B surveillance data available for scientific research,
allowing researchers to study aircraft trajectories, traffic flows, operational
patterns, and anomalous events [1].

A flight trajectory can be viewed as an ordered sequence of aircraft positions
through space and time. Although flights sharing the same origin and destination
serve the same city pair, their actual paths may differ because of route
planning, air traffic control, weather, wind conditions, traffic demand, and
other operational factors.

Unsupervised trajectory clustering therefore provides a method for identifying
recurring operational route patterns without requiring predefined route labels.

### 1.2 Problem Statement

Traditional clustering algorithms operate on fixed-dimensional feature vectors,
whereas complete flight trajectories are sequences of geographic observations.
A meaningful flight-trajectory clustering framework must therefore address two
problems:

1. how to represent and compare complete trajectories; and
2. how to identify recurring groups while allowing unusual trajectories to
   remain unassigned.

Previous studies have demonstrated the usefulness of density-based clustering
and trajectory-distance measures for analysing air traffic flows [2–6].

Kim and Mavris [2], in particular, used a precomputed Hausdorff-distance matrix
with HDBSCAN to cluster historical flight trajectories and to identify both
representative flight cases and irregular trajectories.

This study builds on that general approach while placing additional emphasis on
preprocessing validation, trajectory representation, implementation
verification, sensitivity analysis, and persistent anomaly-candidate
identification.

### 1.3 Research Objectives

The objectives of this study are to:

1. identify recurring operational trajectory families for LAX → JFK flights;
2. select a real observed representative trajectory for each cluster;
3. identify trajectories that do not conform to the main recurring route
   structures;
4. evaluate whether the resulting clustering is robust to reasonable modelling
   choices; and
5. investigate whether persistent and geometrically extreme trajectories
   correspond to identifiable operational or environmental events.

### 1.4 Scope

The present study considers only flights travelling from LAX to JFK.

The primary dataset contains:

- 554 January 2026 trajectories;
- 800 July 2026 trajectories;
- 1,354 trajectories in total.

Multiple airlines are included.

January and July are not assumed to represent universally normal and abnormal
seasons. Month information is retained as metadata for interpretation rather
than being used directly as a clustering feature.

The primary clustering considers horizontal trajectory geometry using latitude
and longitude. Altitude, velocity, weather, aircraft type, and other operational
variables are not included in the primary trajectory-distance metric.

---

## 2. Related Work

Flight-trajectory clustering has been applied to several air-traffic
management problems, including identification of recurring traffic flows,
representative-route selection, operational categorisation, trajectory
prediction, and anomaly detection.

### 2.1 Hausdorff Distance and HDBSCAN for Flight Trajectories

The study most closely related to the present work is that of Kim and Mavris
[2]. Their research proposed an unsupervised trajectory-clustering methodology
to select representative real-world flight cases for offline evaluation of a
Real-Time Trajectory Optimization (RTOP) framework.

Their methodology uses a precomputed distance matrix based on Hausdorff
distance and applies HDBSCAN to group similar flight trajectories. HDBSCAN was
selected because the number of trajectory groups does not need to be specified
in advance and because trajectories outside the dominant density structures can
remain unassigned as noise.

Kim and Mavris considered two main uses of the clustering result. The first was
to reduce a group of similar trajectories to a representative real flight case.
The second was to identify irregular trajectories represented by HDBSCAN noise,
which could provide useful cases for studying non-standard operations.

Their study analysed 491 U.S. domestic flights from selected major airline
markets and subsequently used representative flights to evaluate the RTOP
framework.

The present study adopts the general idea of combining whole-trajectory
Hausdorff distance with HDBSCAN, but the research objective is different.
Rather than using clustering primarily to select simulation-evaluation cases,
the present work focuses on understanding recurring route structure and
identifying persistent geometric anomaly candidates.

### 2.2 Representative Trajectory Selection

Kim and Mavris [2] generated a median representation of the trajectories in
each cluster and selected an observed flight having minimum great-circle
distance to that median trajectory.

In the present study, the representative trajectory is instead defined as the
cluster medoid.

For a cluster $C$, the medoid is the observed trajectory

$$
m = \arg\min_{i \in C}
\frac{1}{|C|-1}
\sum_{j \in C,\; j \neq i}
H(i,j)
$$

where $H(i,j)$ is the same Hausdorff distance used to construct the clustering
distance matrix.

This has two useful properties. First, the representative trajectory is always
a real observed flight. Second, representative-route selection remains within
the same trajectory-distance geometry used by the clustering algorithm rather
than introducing a separate representative-path definition.

### 2.3 Other Flight-Trajectory Clustering Studies

Bolić et al. [3] applied trajectory clustering to historical air-traffic data
to investigate recurring airline routing behaviour and operational
categorisation. Their work demonstrates the usefulness of complete-trajectory
distance measures, including undirected Hausdorff distance, for analysing
historical aircraft trajectories.

Corrado et al. [4] studied trajectory clustering in terminal airspace using
HDBSCAN and demonstrated that clustering performance depends strongly on how
trajectory distance is defined. Their work motivates the need to evaluate
trajectory-distance choices instead of assuming that one distance formulation
is universally appropriate.

This consideration directly influenced the sensitivity analysis in the present
study, where the primary symmetric Hausdorff distance is compared with discrete
Fréchet distance.

### 2.4 Trajectory Anomaly Detection

Anomaly detection from aircraft trajectories requires a distinction between
geometric non-conformance and operational abnormality.

Corrado et al. [5] used clustering-based methods to detect spatial anomalies in
ADS-B trajectory data and subsequently investigated their relationship with
energy anomalies. Their work demonstrates that spatially unusual trajectories
can provide useful anomaly candidates while additional operational variables
are needed for interpretation.

Olive and Basora [6] developed an unsupervised framework for detecting and
characterising significant events in historical aircraft trajectories. Their
results show that unusual trajectory behaviour can arise from different
operational contexts, including poor weather and air-traffic-control actions.

These studies motivate the interpretation adopted here: an HDBSCAN noise label
indicates geometric non-conformance relative to recurring flight patterns, but
it does not by itself identify the physical or operational cause.

### 2.5 Position of the Present Study

The present work therefore does not claim novelty for the basic combination of
Hausdorff distance and HDBSCAN.

Instead, it extends this established trajectory-clustering approach through:

- detailed ADS-B quality validation and harmonisation;
- fixed arc-length trajectory representation;
- manual implementation of the complete HDBSCAN procedure;
- exact validation against an external HDBSCAN implementation;
- HDBSCAN hyperparameter sensitivity analysis;
- full 100-point versus 200-point representation sensitivity;
- full Hausdorff versus discrete Fréchet sensitivity;
- medoid-based representative-route selection;
- persistence analysis of HDBSCAN noise; and
- event-level investigation of temporally concentrated geometric anomalies.

The central research question is therefore not only whether recurring flight
trajectory groups can be identified, but whether those groups and their
associated anomaly candidates remain scientifically meaningful under reasonable
changes in modelling assumptions.

## 3. Dataset

### 3.1 Data Source

The flight data originate from the OpenSky Network [1].

The working dataset consists of individual CSV files in which each file
represents one LAX → JFK flight trajectory.

### 3.2 January and July Data

The final processed dataset contains:

| Month | Trajectories |
|---|---:|
| January 2026 | 554 |
| July 2026 | 800 |
| **Total** | **1,354** |

The two months have different source-data provenance.

January source data were available at approximately one-second temporal
resolution and therefore permitted detailed validation of position freshness,
stale coordinates, discontinuities, and resampling behaviour.

For July, only reduced approximately ten-second source trajectories were
available for the primary clustering dataset.

Because of this difference, direct month comparisons are treated as descriptive
and are not interpreted as evidence of general seasonal behaviour.

### 3.3 Data Quality

ADS-B observations may contain stale positions, repeated coordinates, temporal
gaps, or geographically inconsistent updates.

January validation therefore distinguished among:

- ordinary row-time gaps;
- stale repeated positions;
- gaps between fresh position updates;
- physically inconsistent position jumps; and
- incomplete or fragmented trajectories.

Of 566 January source trajectories, 554 were retained.

For July, 810 candidate trajectories were assessed and 800 were retained after
excluding multisegment or severe spatially inconsistent cases.

The combined processed dataset contains 1,354 complete trajectories.

---

## 4. Methodology

### 4.1 Overall Pipeline

The final workflow is:

```text
ADS-B trajectory data
        ↓
quality validation and cleaning
        ↓
10-second working trajectories
        ↓
arc-length trajectory representation
        ↓
pairwise trajectory distance
        ↓
precomputed distance matrix
        ↓
manual HDBSCAN
        ↓
route clusters + noise candidates
        ↓
representative routes
        ↓
robustness and anomaly analysis
```

### 4.2 Temporal Harmonization

A ten-second working interval was selected after comparing candidate temporal
resolutions.

This interval substantially reduces computational cost while preserving the
major horizontal route geometry required for trajectory clustering.

### 4.3 Arc-Length Representation

Because flights contain different numbers of observations and may have
different durations, trajectories are converted to a fixed geometric
representation based on cumulative travelled arc length.

Candidate resolutions of 25, 50, 100, and 200 points were evaluated.

The 25- and 50-point representations produced substantially larger geometric
approximation errors.

The 100-point representation was selected as the primary configuration because
it provided strong geometric fidelity while remaining much cheaper to compare
than the 200-point representation.

### 4.4 Haversine Point Distance

Latitude and longitude lie on the Earth's surface and should not be treated as
ordinary Cartesian coordinates over long continental distances.

Point-to-point separation is therefore computed using the Haversine
great-circle formulation.

### 4.5 Symmetric Hausdorff Trajectory Distance

Let trajectories $P$ and $Q$ consist of sets of geographic points.

The directed Hausdorff distance from $P$ to $Q$ is

$$
h(P,Q) = \max_{p \in P} \min_{q \in Q} d(p,q)
$$

where $d(p,q)$ is the Haversine distance.

The symmetric Hausdorff distance is

$$
H(P,Q) = \max\left(h(P,Q), h(Q,P)\right)
$$

The distance provides a single geometric dissimilarity value for every pair of
complete flight trajectories.

Hausdorff distance has previously been used for aviation trajectory clustering
[2,3].

For $N = 1{,}354$ flights, the final distance matrix has dimension

$$
1354 \times 1354
$$

There are 915,981 unique trajectory pairs.

### 4.6 HDBSCAN

HDBSCAN was chosen because it does not require the number of clusters to be
specified in advance and can leave observations outside dense structures as
noise [2,7,8].

The algorithm was implemented manually using:

```text
core distances
        ↓
mutual-reachability distances
        ↓
minimum spanning tree
        ↓
single-linkage hierarchy
        ↓
condensed cluster hierarchy
        ↓
cluster stability
        ↓
Excess of Mass selection
        ↓
final labels
```

The final parameters are:

```text
min_samples = 15
min_cluster_size = 30
```

The manual `min_samples = 15` convention counts the fifteen nearest *other*
trajectories. For validation, this corresponds to
`sklearn.cluster.HDBSCAN(min_samples=16)` because the reference implementation
counts the observation itself.

### 4.7 Representative Route Selection

A medoid is selected for each final cluster.

For a cluster $C$, the medoid is the observed trajectory

$$
m = \arg\min_{i \in C}
\frac{1}{|C|-1}
\sum_{j \in C,\; j \neq i}
H(i,j)
$$

This guarantees that the representative path corresponds to a real observed
flight while remaining central according to the same Hausdorff geometry used
for clustering.

### 4.8 Validation Metrics

Several complementary validation measures are used.

Silhouette analysis evaluates cohesion and separation within the selected
clusters [9].

Adjusted Rand Index is used to compare complete clustering partitions across
sensitivity configurations [10].

For noise sets, Jaccard similarity and direct noise-mask agreement are also
reported because ARI alone does not isolate noise-membership changes.

---

## 5. Results

### 5.1 Final Clustering

The primary HDBSCAN result contains four clusters.

| Group | Number of flights |
|---|---:|
| Cluster 0 | 73 |
| Cluster 1 | 558 |
| Cluster 2 | 88 |
| Cluster 3 | 340 |
| Noise | 295 |

Therefore:

```text
Clustered trajectories: 1,059 / 1,354 = 78.213%
Noise trajectories:       295 / 1,354 = 21.787%
```

### 5.2 Representative Routes

The medoid trajectories are:

| Cluster | Representative trajectory |
|---|---|
| 0 | `20260116T220252Z_adde95_JBU1124` |
| 1 | `20260111T074405Z_adabfe_JBU324` |
| 2 | `20260112T064839Z_a33328_AAL10` |
| 3 | `20260131T200523Z_a03765_AAL32` |

These routes provide observed examples of the four recurring geometric route
families.

### 5.3 Cluster Separation

Pairwise Hausdorff distances among medoids show that the closest representative
pair is:

```text
Cluster 2 ↔ Cluster 3
213.922 km
```

The most separated pair is:

```text
Cluster 0 ↔ Cluster 1
634.983 km
```

### 5.4 Silhouette Evaluation

The overall silhouette score among the 1,059 clustered trajectories is:

```text
0.490665
```

Mean silhouette by cluster is:

```text
Cluster 0: 0.561794
Cluster 1: 0.541886
Cluster 2: 0.388803
Cluster 3: 0.417695
```

Only 29 clustered trajectories have negative silhouette scores.

Silhouette was used as complementary evaluation evidence and was not used to
tune the final HDBSCAN configuration.

### 5.5 Month Composition

January contains:

```text
89 noise trajectories out of 554 = 16.06%
```

July contains:

```text
206 noise trajectories out of 800 = 25.75%
```

This difference is descriptive only.

It does not establish a seasonal effect because the analysis covers only two
months of one year and the two months have different source-data provenance.

---

## 6. Robustness and Sensitivity Analysis

### 6.1 Manual Implementation Validation

The manual implementation was compared with scikit-learn's HDBSCAN using the
equivalent neighbour-counting convention.

The result was exact:

```text
ARI, all trajectories:       1.000000
ARI, commonly clustered:     1.000000
Noise-set Jaccard:            1.000000
Noise-mask agreement:         1.000000
```

All 1,354 trajectories received exactly the same cluster/noise membership.

### 6.2 Representation Sensitivity

The complete clustering was repeated using 200 arc-length points per
trajectory.

Comparison against the primary 100-point result produced:

```text
Overall ARI:            0.989996
Common-cluster ARI:     1.000000
Noise-set Jaccard:      0.979730
Noise-mask agreement:   0.995569
```

Only six trajectories changed cluster/noise status.

This indicates that the four principal route families are effectively unchanged
by doubling representation resolution.

### 6.3 Trajectory-Metric Sensitivity

Discrete Fréchet distance [11] was evaluated as an alternative whole-trajectory
metric.

Across all trajectory pairs, Hausdorff and discrete Fréchet distances were
highly correlated.

The final HDBSCAN comparison produced:

```text
Overall ARI:            0.993846
Common-cluster ARI:     1.000000
Noise-set Jaccard:      0.983108
Noise-mask agreement:   0.996307
```

Only five trajectories changed cluster/noise status.

The principal clustering structure is therefore not dependent on Hausdorff
distance alone.

Hausdorff was retained because it produced essentially the same scientific
partition at substantially lower computational cost.

### 6.4 HDBSCAN Parameter Sensitivity

Neighboring `min_samples` and `min_cluster_size` configurations were evaluated.

The major route families remained stable across reasonable parameter changes,
although trajectories close to cluster boundaries could move between a cluster
and noise.

This distinction is important because it indicates that most sensitivity lies
in borderline membership rather than in the existence of the major route
families themselves.

---

## 7. Anomaly-Candidate Analysis

### 7.1 HDBSCAN Noise

The primary configuration identifies:

```text
295 noise trajectories
```

These are interpreted as anomaly candidates, not automatically abnormal
operations.

A trajectory may be labelled noise because of:

- a legitimate but rare routing;
- weather-related deviation;
- ATC intervention;
- operational constraints;
- diversion or rerouting;
- incomplete or low-quality ADS-B observations; or
- other causes.

### 7.2 Noise Persistence

Noise membership was compared across three neighboring
`min_samples` configurations.

Results:

```text
Noise in exactly 1/3 configurations:   0
Noise in exactly 2/3 configurations:  56
Noise in all 3/3 configurations:     239
```

Therefore approximately 81% of the primary noise trajectories remain noise in
all three tested configurations.

This provides stronger evidence that many noise cases are genuine geometric
departures from the recurring route structures rather than artifacts of a
single hyperparameter value.

### 7.3 Geometric Extremeness

For every primary noise trajectory, distance to the nearest cluster medoid was
computed.

```text
Minimum:  92.486 km
Median:  201.418 km
P95:     443.453 km
Maximum: 799.939 km
```

The 95th percentile was used descriptively to identify the most geometrically
extreme primary noise candidates.

It was not used to tune HDBSCAN.

### 7.4 July 28, 2026 Concentration

A notable temporal concentration was identified after clustering.

Across the complete dataset, 15 primary noise trajectories exceeded the 95th
percentile of nearest-medoid distance.

Eight of these 15 trajectories occurred on 28 July 2026.

The surrounding dates show:

```text
27 July:
29 total flights
16 noise
14 persistent noise
2 geometrically extreme

28 July:
20 total flights
14 noise
14 persistent noise
8 geometrically extreme

29 July:
26 total flights
11 noise
9 persistent noise
0 geometrically extreme
```

July 28 therefore represents only:

```text
20 / 1,354 = 1.48%
```

of the complete study dataset but contains:

```text
8 / 15 = 53.3%
```

of the most geometrically extreme primary noise trajectories.

This signal was discovered by the unsupervised clustering process rather than
by selecting the date in advance.

### 7.5 Current Interpretation

The current supported statement is:

> The unsupervised clustering independently identified an unusually
> concentrated set of persistent and geometrically extreme trajectories on
> July 28, 2026. The cause has not yet been established.

Higher-resolution July ADS-B data and independent meteorological or operational
evidence are required before attributing these deviations to weather, air
traffic control, or another cause.

This interpretation follows the broader anomaly-detection literature, where
trajectory deviations are identified first and then contextualized using
independent operational evidence [5,6].

---

## 8. Discussion

### 8.1 Route Structure

The clustering identifies four recurring geometric route families between LAX
and JFK.

The high agreement observed across representation and trajectory-distance
sensitivity tests suggests that these structures are persistent features of the
dataset rather than artifacts of one modelling decision.

### 8.2 Relationship to Previous Research

The overall clustering concept is closely related to Kim and Mavris [2], who
also used Hausdorff trajectory distances, a precomputed distance matrix, and
HDBSCAN to identify representative and anomalous flight cases.

The current study should therefore not claim novelty for the basic
Hausdorff–HDBSCAN combination.

Instead, its contribution lies in:

- detailed ADS-B quality validation;
- fixed arc-length trajectory representation;
- manual HDBSCAN implementation and exact external validation;
- full representation-resolution sensitivity;
- full Hausdorff-versus-Fréchet sensitivity;
- parameter sensitivity;
- medoid-based representative routes;
- persistent-noise analysis; and
- event-level investigation of concentrated geometric deviations.

### 8.3 Interpretation of Noise

A central methodological principle of the study is that HDBSCAN noise and
operational anomaly are not synonymous.

Density-based clustering identifies geometric non-conformance relative to the
observed population.

Operational interpretation requires additional evidence.

This distinction prevents rare but valid routes or data-quality problems from
being incorrectly labelled as hazardous operations.

### 8.4 Limitations

The present study has several limitations.

First, only one origin-destination pair is studied.

Second, January and July have different source-data provenance.

Third, clustering considers horizontal geometry only.

Fourth, HDBSCAN noise labels indicate geometric non-conformance rather than a
known operational cause.

Fifth, the July 28 event has not yet been validated using independent
meteorological or ATC information.

Finally, results from two months of one year should not be generalized to all
seasonal operations.

---

## 9. Conclusion

This study developed and validated a complete trajectory-clustering framework
for 1,354 LAX → JFK flights.

Using 100-point arc-length representations, Haversine geographic distance,
symmetric Hausdorff trajectory distance, and a manually implemented HDBSCAN
algorithm, four recurring route families were identified together with 295
noise trajectories.

The clustering result is strongly supported by independent robustness tests.
The manual implementation exactly reproduces the equivalent scikit-learn
partition. Doubling trajectory representation resolution produces an ARI of
0.989996, while replacing Hausdorff distance with discrete Fréchet distance
produces an ARI of 0.993846. In both experiments, the memberships of
trajectories clustered under both configurations are identical.

These results indicate that the major route structures are highly stable and
that modelling sensitivity is concentrated primarily among cluster/noise
boundary trajectories.

The anomaly analysis further identifies a persistent set of geometrically
unusual flights and reveals an especially strong concentration of extreme
trajectories on July 28, 2026.

The cause of this concentration remains unresolved.

The next research stage is therefore to validate the detected event using
higher-resolution ADS-B and independent weather or operational evidence.

---

## 10. Future Work

Future work will focus on:

1. event-level validation of July 27–29 using higher-resolution ADS-B;
2. comparison with convective weather and other meteorological information;
3. investigation of possible ATC or operational explanations;
4. inclusion of additional origin-destination pairs;
5. extension to additional months and years;
6. investigation of altitude and other trajectory-state variables;
7. comparison with additional trajectory-clustering baselines; and
8. evaluation of whether persistent noise can be transformed into a calibrated
   anomaly score.

---

## References

[1] M. Schäfer, M. Strohmeier, V. Lenders, I. Martinovic, and M. Wilhelm,
“Bringing Up OpenSky: A Large-scale ADS-B Sensor Network for Research,”
Proceedings of the 13th IEEE/ACM International Symposium on Information
Processing in Sensor Networks, pp. 83–94, 2014.
doi: 10.1109/IPSN.2014.6846743.

[2] J. Kim and D. N. Mavris,
“Flight data clustering for offline evaluation of real-time trajectory
optimization framework,”
Decision Analytics Journal, vol. 7, 100221, 2023.
doi: 10.1016/j.dajour.2023.100221.

[3] T. Bolić, L. Castelli, A. De Lorenzo, and F. Vascotto,
“Trajectory Clustering for Air Traffic Categorisation,”
Aerospace, vol. 9, no. 5, 227, 2022.
doi: 10.3390/aerospace9050227.

[4] S. J. Corrado, T. G. Puranik, O. J. Pinon, and D. N. Mavris,
“Trajectory Clustering within the Terminal Airspace Utilizing a Weighted
Distance Function,”
Proceedings, vol. 59, no. 1, 7, 2020.
doi: 10.3390/proceedings2020059007.

[5] S. J. Corrado, T. G. Puranik, O. Pinon Fischer, and D. N. Mavris,
“A clustering-based quantitative analysis of the interdependent relationship
between spatial and energy anomalies in ADS-B trajectory data,”
Transportation Research Part C: Emerging Technologies, vol. 131, 103331,
2021.
doi: 10.1016/j.trc.2021.103331.

[6] X. Olive and L. Basora,
“Detection and identification of significant events in historical aircraft
trajectory data,”
Transportation Research Part C: Emerging Technologies, vol. 119, 102737,
2020.
doi: 10.1016/j.trc.2020.102737.

[7] R. J. G. B. Campello, D. Moulavi, and J. Sander,
“Density-based clustering based on hierarchical density estimates,”
Pacific-Asia Conference on Knowledge Discovery and Data Mining,
pp. 160–172, 2013.
doi: 10.1007/978-3-642-37456-2_14.

[8] R. J. G. B. Campello, D. Moulavi, A. Zimek, and J. Sander,
“Hierarchical density estimates for data clustering, visualization, and
outlier detection,”
ACM Transactions on Knowledge Discovery from Data,
vol. 10, no. 1, Article 5, 2015.
doi: 10.1145/2733381.

[9] P. J. Rousseeuw,
“Silhouettes: A graphical aid to the interpretation and validation of cluster
analysis,”
Journal of Computational and Applied Mathematics,
vol. 20, pp. 53–65, 1987.
doi: 10.1016/0377-0427(87)90125-7.

[10] L. Hubert and P. Arabie,
“Comparing partitions,”
Journal of Classification,
vol. 2, pp. 193–218, 1985.
doi: 10.1007/BF01908075.

[11] T. Eiter and H. Mannila,
“Computing Discrete Fréchet Distance,”
Technical Report CD-TR 94/64,
Technical University of Vienna, 1994.