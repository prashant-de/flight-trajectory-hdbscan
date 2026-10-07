# Clustering Flight Trajectories Using HDBSCAN: Identifying Operational Patterns and Anomaly Candidates
## Abstract

Aircraft trajectory data provide a detailed record of real-world flight
operations and can be used to identify recurring route structures as well as
flights that deviate from common operational patterns. This study develops and
evaluates an unsupervised trajectory-clustering framework for flights operating
from Los Angeles International Airport (LAX) to John F. Kennedy International
Airport (JFK). The dataset contains 1,354 complete trajectories from January
and July 2026.

Each complete flight is treated as one clustering sample. Following data
validation and temporal harmonisation, trajectories are represented using 100
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
239 remain noise across three neighbouring  HDBSCAN parameter configurations.
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

## 3. Dataset and Data Provenance

### 3.1 Data Source and Study Population

The trajectory data used in this study originate from the OpenSky Network,
which provides large-scale ADS-B surveillance data for aviation research [1].

The study focuses on flights operating in a single direction between:

- Los Angeles International Airport (LAX), and
- John F. Kennedy International Airport (JFK).

One complete flight trajectory is treated as one observation for clustering.
Individual ADS-B position records are therefore not independent clustering
samples; they are sequential measurements describing a single flight.

Multiple airlines are included in the dataset.

The final study population contains trajectories from two selected months in
2026:

| Month | Candidate Flights | Retained Flights |
|---|---:|---:|
| January 2026 | 566 | 554 |
| July 2026 | 810 | 800 |
| **Total** | **1,376** | **1,354** |

The final clustering dataset therefore contains **1,354 complete LAX → JFK
flight trajectories**.

---

### 3.2 January Data Provenance

For January 2026, the original source files were available at approximately
one-second temporal resolution.

These higher-resolution records contained additional state and data-quality
information beyond latitude and longitude and allowed detailed examination of
issues such as:

- stale or repeatedly reported aircraft positions;
- temporal gaps;
- discontinuities between fresh position updates;
- physically implausible spatial jumps; and
- fragmented or multisegment trajectories.

This made January the primary month used to develop and validate the
preprocessing procedure.

Of 566 January candidate trajectories, 12 were excluded as severe
multisegment cases.

The remaining 554 trajectories were cleaned and converted into the common
working representation used by the clustering pipeline.

Across these retained January flights, the original high-resolution data
contained approximately 9.42 million observations before cleaning.

The preprocessing procedure removed observations that could not be used
reliably for trajectory geometry, including unavailable positions, stale
carried-forward coordinates, and physically inconsistent position updates.

After cleaning, approximately 8.41 million high-resolution January
observations remained before temporal harmonisation.

---

### 3.3 July Data Provenance

The July 2026 data were available in a different form.

The original high-resolution July ADS-B files were not available for the
primary clustering study. Instead, the available July source files had already
been reduced to approximately 10-second temporal resolution.

These reduced files retained fields including:

```text
time
lat
lon
flight_key
velocity
baroaltitude
alert
icao24
```

Because the original higher-resolution observations were unavailable, the same
fresh-position validation that was possible for January could not be reproduced
fully for July.

July trajectories were nevertheless subjected to the quality checks that could
be supported by the available data, including trajectory segmentation and
spatial-consistency checks.

Of 810 candidate July trajectories:

```text
5 multisegment trajectories were excluded
5 severe spatially inconsistent trajectories were excluded
```

This resulted in **800 retained July trajectories**.

The difference in source-data provenance between January and July is explicitly
preserved as a limitation of the study.

---

### 3.4 Data Cleaning and Spatial Consistency

Trajectory quality is particularly important for distance-based clustering
because a small number of incorrect position observations can strongly affect
whole-trajectory distances.

The preprocessing procedure therefore distinguishes between several different
types of data-quality problems rather than treating every temporal gap as
equivalent.

These include:

- missing or unusable positions;
- repeated stale positions;
- gaps between observations;
- gaps between genuinely updated positions;
- spatial discontinuities; and
- severe trajectory fragmentation.

For spatial consistency checking, an update was considered suspicious when it
simultaneously implied:

```text
position displacement > 10 km
```

and

```text
implied ground speed > 3000 km/h
```

Using both conditions prevents an ordinary large displacement over a realistic
time interval from being rejected solely because of distance.

A complete flight was excluded when more than 20% of its unique position
updates were rejected as spatially inconsistent.

These thresholds were used as data-quality safeguards rather than as
definitions of operational anomaly.

---

### 3.5 Temporal Harmonisation

The January source data were substantially denser than the July source data.
A common temporal working resolution was therefore required before combining
the two months.

Candidate working intervals of approximately:

```text
5 seconds
10 seconds
30 seconds
```

were evaluated during preprocessing.

A **10-second working interval** was selected as the primary configuration.

This resolution provided a practical compromise between:

- preservation of the large-scale horizontal route geometry;
- consistency with the available July data;
- computational cost; and
- dataset size.

After preprocessing, January produced:

```text
554 trajectories
915,582 trajectory observations
```

July produced:

```text
800 trajectories
1,394,558 trajectory observations
```

The final harmonised dataset therefore contains:

```text
1,354 trajectories
2,310,140 trajectory observations
```

At this stage, each processed trajectory follows a common schema containing:

```text
sample_index
sample_time
time_utc
lat
lon
interpolated
```

This harmonised dataset forms the input to the geometric trajectory
representation described in Section 4.

---

### 3.6 Month as Metadata Rather Than a Clustering Feature

January and July labels are preserved for post-clustering interpretation but
are not included in the trajectory-distance calculation or HDBSCAN input.

Consequently, the clustering algorithm does not know which month a trajectory
belongs to.

Any difference observed later between January and July therefore emerges from
the geometric clustering result rather than from explicitly providing month
information to the algorithm.

However, such differences must be interpreted cautiously.

The study contains only two selected months from one year, and the two months
have different source-data provenance. Therefore, differences in cluster or
noise proportions are treated as **descriptive observations**, not as evidence
of a general winter-versus-summer or seasonal effect.

---

### 3.7 Dataset Limitations

Several limitations of the dataset should be considered when interpreting the
results.

First, the study covers only one origin-destination pair.

Second, January and July have different source temporal resolutions and
available quality fields.

Third, the July high-resolution source observations were not available for the
primary analysis.

Fourth, the primary clustering uses horizontal trajectory geometry only.
Altitude, velocity, aircraft type, meteorological conditions, and air traffic
control information are not included directly in the clustering distance.

Finally, ADS-B trajectory data describe observed aircraft movement but do not
independently explain why a particular route was flown.

These limitations are particularly important when interpreting HDBSCAN noise
trajectories as anomaly candidates.

## 4. Methodology

### 4.1 Overall Analysis Pipeline

The complete analysis pipeline is shown conceptually below:

```text
LAX → JFK ADS-B trajectories
        ↓
data-quality validation and cleaning
        ↓
10-second harmonised working trajectories
        ↓
fixed-length geometric trajectory representation
        ↓
pairwise whole-trajectory distance calculation
        ↓
1,354 × 1,354 precomputed distance matrix
        ↓
manual HDBSCAN
        ↓
recurring route clusters + noise trajectories
        ↓
representative route medoids
        ↓
robustness and anomaly-candidate analysis
```

A fundamental design decision is that **one complete flight trajectory is one
HDBSCAN sample**.

The individual ADS-B observations within a flight are used to describe its
geometry but are not treated as independent clustering samples.

Therefore, for $N$ flights, the clustering problem contains $N$ samples rather
than the total number of ADS-B position records.

For the final dataset:

$$
N = 1{,}354
$$

and the resulting trajectory-distance matrix has dimension:

$$
1354 \times 1354
$$

---

### 4.2 Harmonised Working Trajectories

The output of the preprocessing stage is a common approximately 10-second
trajectory representation for both January and July.

Each processed trajectory contains:

```text
sample_index
sample_time
time_utc
lat
lon
interpolated
```

These observations preserve the chronological movement of each aircraft from
LAX toward JFK.

The 10-second working interval is a preprocessing representation rather than
the final representation used directly for pairwise trajectory comparison.

Because flights have different durations and therefore different numbers of
10-second observations, an additional geometric representation step is required
before systematic trajectory-distance analysis.

---

### 4.3 Fixed-Length Arc-Length Representation

Aircraft flying between the same airports may have different flight durations,
sampling counts, and speeds.

Directly comparing observations by their row index would therefore incorrectly
assume that corresponding row positions represent equivalent locations along
the route.

To avoid this problem, each flight is represented according to cumulative
horizontal distance travelled along the trajectory.

For consecutive trajectory points, geographic segment lengths are computed
using great-circle distance.

The cumulative path length is then normalised from the beginning to the end of
the trajectory.

New points are interpolated at equally spaced positions along this cumulative
arc length.

Conceptually:

```text
original processed trajectory
        ↓
calculate segment distances
        ↓
calculate cumulative travelled distance
        ↓
normalise cumulative arc length
        ↓
sample equally spaced positions along the route
        ↓
fixed-length trajectory representation
```

Candidate representations containing:

```text
25 points
50 points
100 points
200 points
```

were evaluated.

The 25-point and 50-point representations produced noticeably larger geometric
approximation errors.

The 100-point representation was selected as the primary configuration because
it preserved the large-scale route geometry while substantially reducing the
computational cost of pairwise trajectory comparison.

The 200-point representation was retained for full sensitivity analysis.

Therefore, each complete flight becomes a geometric object containing:

$$
100
$$

ordered latitude-longitude positions.

The 100 positions together describe **one HDBSCAN sample**; they are not 100
independent samples.

---

### 4.4 Haversine Point-to-Point Distance

Latitude and longitude are geographic coordinates on the Earth's surface and
should not be treated as ordinary Cartesian coordinates for continental-scale
distance calculations.

The Haversine formulation is therefore used to compute the great-circle
distance between individual trajectory points.

For two geographic positions

$$
p_1 = (\phi_1,\lambda_1)
$$

and

$$
p_2 = (\phi_2,\lambda_2),
$$

where $\phi$ denotes latitude and $\lambda$ denotes longitude in radians, the
Haversine formulation is:

$$
a =
\sin^2\left(\frac{\Delta\phi}{2}\right)
+
\cos(\phi_1)\cos(\phi_2)
\sin^2\left(\frac{\Delta\lambda}{2}\right)
$$

$$
c =
2\arctan2\left(\sqrt{a},\sqrt{1-a}\right)
$$

and the great-circle distance is:

$$
d(p_1,p_2) = R c
$$

where $R$ is the Earth's radius.

This point-level geographic distance is subsequently used inside the
whole-trajectory distance calculation.

---

### 4.5 Symmetric Hausdorff Trajectory Distance

Haversine distance compares two individual geographic positions. A separate
distance definition is required to compare two complete flight trajectories.

The primary trajectory-level metric in this study is symmetric Hausdorff
distance.

Let trajectories $P$ and $Q$ contain sets of geographic points.

The directed Hausdorff distance from $P$ to $Q$ is:

$$
h(P,Q)
=
\max_{p \in P}
\min_{q \in Q}
d(p,q)
$$

where $d(p,q)$ is the Haversine distance between two geographic points.

Similarly:

$$
h(Q,P)
=
\max_{q \in Q}
\min_{p \in P}
d(q,p)
$$

The symmetric Hausdorff distance is:

$$
H(P,Q)
=
\max\left(h(P,Q),h(Q,P)\right)
$$

This produces one distance value, measured in kilometres, for every pair of
complete flight trajectories.

The interpretation is geometric: two trajectories have a small Hausdorff
distance when their routes remain spatially close, while a trajectory
containing substantial route deviation can have a larger distance from another
trajectory.

Hausdorff distance does not require the two source trajectories to contain the
same number of original ADS-B observations.

For the final $N=1{,}354$ trajectories, the number of unique trajectory pairs
is:

$$
\frac{N(N-1)}{2}
=
\frac{1354 \times 1353}{2}
=
915{,}981
$$

The pairwise distances are stored in a symmetric matrix:

$$
D \in \mathbb{R}^{1354 \times 1354}
$$

where:

$$
D_{ij} = H(T_i,T_j)
$$

and:

$$
D_{ii} = 0.
$$

The flight ordering associated with the matrix is stored separately so that
every matrix row and column remains traceable to its original trajectory.

---

### 4.6 Alternative Trajectory Metric

Discrete Fréchet distance was also implemented as an alternative
whole-trajectory distance measure [11].

Unlike Hausdorff distance, which compares spatial sets, discrete Fréchet
distance considers the ordering of positions along the two trajectories.

It therefore provides a useful sensitivity test for determining whether the
identified route structure depends strongly on the choice of trajectory
distance.

Discrete Fréchet distance is not used in the primary model.

Instead, the entire clustering procedure is repeated with the Fréchet distance
matrix and compared with the primary Hausdorff result.

This allows the stability of the scientific conclusions to be tested rather
than assuming that Hausdorff distance is uniquely appropriate.

---

### 4.7 HDBSCAN

Hierarchical Density-Based Spatial Clustering of Applications with Noise
(HDBSCAN) is used to identify recurring route structures from the precomputed
trajectory-distance matrix [7,8].

HDBSCAN is suitable for this problem because:

- the number of route families does not need to be specified in advance;
- clusters can contain different numbers of trajectories;
- density structure is considered hierarchically; and
- trajectories that do not belong strongly to recurring dense structures can
  remain unassigned as noise.

The complete HDBSCAN procedure was implemented manually rather than using a
clustering library for the primary result.

The implementation follows the sequence:

```text
trajectory-distance matrix
        ↓
core distances
        ↓
mutual-reachability distances
        ↓
minimum spanning tree
        ↓
single-linkage hierarchy
        ↓
condensed hierarchy
        ↓
cluster stability
        ↓
Excess of Mass cluster selection
        ↓
final cluster and noise labels
```

The main components are described below.

---

### 4.8 Core Distance

For each trajectory $x_i$, the core distance is determined from its neighboring
trajectories in the precomputed distance matrix.

Using the manual implementation convention adopted in this study:

$$
\operatorname{core}_k(x_i)
$$

is the distance from trajectory $x_i$ to its $k$-th nearest **other**
trajectory.

The primary configuration uses:

```text
min_samples = 15
```

Therefore, the core distance of each flight is determined by its fifteenth
nearest other flight.

A smaller core distance indicates that the trajectory lies in a relatively
dense region of trajectory space, while a larger value indicates a more
isolated trajectory.

---

### 4.9 Mutual-Reachability Distance

HDBSCAN transforms the original pairwise trajectory distances into
mutual-reachability distances.

For trajectories $x_i$ and $x_j$:

$$
d_{\mathrm{mreach}}(x_i,x_j)
=
\max
\left(
\operatorname{core}_k(x_i),
\operatorname{core}_k(x_j),
d(x_i,x_j)
\right)
$$

where $d(x_i,x_j)$ is the original Hausdorff trajectory distance.

This transformation reduces the influence of isolated pairwise connections and
incorporates the local density surrounding both trajectories.

The resulting complete mutual-reachability graph forms the basis of the
hierarchical clustering stage.

---

### 4.10 Minimum Spanning Tree and Hierarchy

A minimum spanning tree (MST) is constructed from the mutual-reachability
distances.

The MST provides a compact representation of the connectivity structure among
all trajectories while preserving the edges required for hierarchical
single-linkage clustering.

Edges are considered according to increasing mutual-reachability distance.

Equivalently, HDBSCAN commonly expresses hierarchy levels using:

$$
\lambda = \frac{1}{d_{\mathrm{mreach}}}
$$

Higher values of $\lambda$ correspond to connections that persist at higher
density levels.

By progressively separating the MST, a hierarchy of candidate trajectory
groups is obtained.

---

### 4.11 Condensed Hierarchy

The ordinary single-linkage hierarchy contains branches of many different
sizes, including very small groups that are not considered meaningful clusters.

HDBSCAN therefore condenses the hierarchy using the parameter:

```text
min_cluster_size = 30
```

A branch must contain at least 30 trajectories to qualify as a candidate
cluster.

When a hierarchy branch divides, sufficiently large child branches can become
new candidate clusters. Small branches do not automatically become final
clusters and may ultimately contribute to noise membership depending on the
subsequent hierarchy and cluster-selection process.

This hierarchical treatment differs from simply applying a fixed distance
threshold.

---

### 4.12 Cluster Stability and Selection

Candidate clusters can exist over different ranges of the HDBSCAN density
hierarchy.

HDBSCAN evaluates cluster persistence through a stability measure based on how
long trajectories remain associated with a cluster as the hierarchy changes.

More persistent branches receive greater stability than structures that appear
only over a narrow density range.

Final clusters are selected using the **Excess of Mass (EOM)** strategy.

Conceptually, EOM compares the stability associated with a parent cluster
against the combined stability of its qualifying descendants.

The hierarchy is therefore reduced to a non-overlapping set of clusters that
best preserves stable density structure.

Trajectories that are not retained as members of the selected clusters receive
the HDBSCAN noise label:

```text
-1
```

In this study, a noise label is interpreted only as **geometric
non-conformance with the selected recurring route structures**.

It is not automatically interpreted as an unsafe, invalid, or abnormal flight.

---

### 4.13 Primary HDBSCAN Configuration

The primary clustering configuration is:

| Component | Primary Choice |
|---|---|
| Clustering object | One complete flight |
| Working temporal resolution | Approximately 10 seconds |
| Geometric representation | 100 arc-length points |
| Point distance | Haversine great-circle distance |
| Trajectory distance | Symmetric Hausdorff distance |
| HDBSCAN input | Precomputed distance matrix |
| `min_samples` | 15 nearest other trajectories |
| `min_cluster_size` | 30 trajectories |
| Cluster selection | Excess of Mass |

The parameter configuration was evaluated together with neighboring parameter
settings.

The final configuration was not selected for the purpose of maximizing the
number of noise trajectories or producing a desired anomaly result.

Parameter robustness is analysed separately in Section 6.

---

### 4.14 Representative Route Selection

After clustering, one representative observed flight is selected from each
cluster.

Rather than constructing an artificial average trajectory, the representative
flight is defined as the cluster medoid.

For a cluster $C$, the medoid is:

$$
m =
\arg\min_{i \in C}
\frac{1}{|C|-1}
\sum_{j \in C,\;j\neq i}
H(i,j)
$$

where $H(i,j)$ is the same Hausdorff trajectory distance used during
clustering.

The medoid therefore minimizes its mean distance to the other members of the
cluster.

This choice has two advantages:

- the representative route is an actual observed flight; and
- representative-route selection remains consistent with the geometric
  distance space used by HDBSCAN.

---

### 4.15 External Implementation Validation

Because HDBSCAN was implemented manually, an independent reference
implementation was used to verify the final result.

The reference comparison uses `sklearn.cluster.HDBSCAN` with the same
precomputed trajectory-distance matrix.

An implementation-convention difference must be accounted for.

The manual implementation defines:

```text
min_samples = 15
```

as the fifteenth nearest **other** trajectory.

The scikit-learn implementation includes the sample itself in its
`min_samples` convention.

Therefore, the equivalent reference configuration is:

```text
manual implementation:
min_samples = 15

scikit-learn reference:
min_samples = 16
```

The comparison evaluates:

- complete partition agreement;
- Adjusted Rand Index;
- agreement among commonly clustered trajectories;
- noise-set Jaccard similarity; and
- noise-mask agreement.

The purpose of this comparison is verification of the manual implementation,
not replacement of the manually implemented algorithm.

---

### 4.16 Robustness Analysis

The clustering result is evaluated under several modelling changes.

The principal sensitivity analyses are:

```text
HDBSCAN parameter sensitivity
        ↓
100-point vs 200-point representation sensitivity
        ↓
Hausdorff vs discrete Fréchet distance sensitivity
        ↓
manual vs scikit-learn implementation validation
```

These experiments distinguish two different concepts:

**HDBSCAN stability** refers to persistence of candidate clusters within the
internal HDBSCAN hierarchy.

**Model robustness** refers to whether the final scientific clustering remains
similar when reasonable methodological choices are changed.

Adjusted Rand Index is used to compare complete partitions [10].

For two clusterings, an ARI close to:

$$
1
$$

indicates very strong agreement.

Because anomaly investigation depends especially on noise membership, the
noise sets are additionally compared using Jaccard similarity:

$$
J(A,B)
=
\frac{|A \cap B|}{|A \cup B|}
$$

where $A$ and $B$ represent the sets of trajectories labelled as noise in two
different configurations.

Noise-mask agreement is also reported.

---

### 4.17 Anomaly-Candidate Framework

The anomaly analysis begins **after** the main unsupervised clustering is
completed.

No known event date, weather information, or predefined anomaly label is
provided to HDBSCAN.

Primary HDBSCAN noise trajectories are ranked and investigated using additional
post-clustering evidence.

Three ideas are considered:

1. whether the trajectory remains noise under nearby HDBSCAN parameter
   configurations;
2. how far the trajectory lies from the nearest representative route medoid;
   and
3. whether unusual trajectories exhibit temporal concentration on particular
   dates.

Persistence across neighboring HDBSCAN settings is used to distinguish
parameter-sensitive boundary cases from trajectories that remain consistently
outside the major route structures.

Nearest-medoid distance provides a geometric measure of how far a noise
trajectory lies from the closest representative route family.

These quantities are used for **post-clustering investigation only**.

They are not used to modify the HDBSCAN labels or tune the primary model.

Independent high-resolution ADS-B, meteorological, or operational evidence is
required before assigning a physical cause to an anomaly candidate.

## 5. Primary Clustering Results

### 5.1 Final HDBSCAN Partition

The primary clustering configuration used:

```text
Trajectory representation: 100 arc-length points
Point distance:             Haversine distance
Trajectory distance:        Symmetric Hausdorff distance
min_samples:                15
min_cluster_size:           30
Cluster selection:          Excess of Mass
```

Applying this configuration to the complete dataset of 1,354 LAX → JFK
trajectories produced four recurring trajectory clusters together with a set
of HDBSCAN noise trajectories.

| Group | Number of Flights | Percentage of Dataset |
|---|---:|---:|
| Cluster 0 | 73 | 5.39% |
| Cluster 1 | 558 | 41.21% |
| Cluster 2 | 88 | 6.50% |
| Cluster 3 | 340 | 25.11% |
| Noise | 295 | 21.79% |
| **Total** | **1,354** | **100.00%** |

The four selected clusters contain:

```text
1,059 trajectories
```

or approximately:

```text
78.21% of the complete dataset
```

The remaining:

```text
295 trajectories
```

or approximately:

```text
21.79%
```

are labelled as HDBSCAN noise.

At this stage, the noise label is interpreted only as indicating that a
trajectory was not retained as part of one of the four recurring density-based
route structures.

It is not interpreted as evidence that the flight was operationally abnormal.

---

### 5.2 Geographic Structure of the Clusters

The final trajectory partition is shown below.

![Final HDBSCAN route clusters](docs/figures/final_route_clusters_map.png)

*Figure 1. Final HDBSCAN clustering of all 1,354 LAX → JFK trajectories.
Cluster colours represent the four recurring route families, while grey
trajectories represent HDBSCAN noise.*

The map shows that the clustering result corresponds to geographically
distinguishable route structures rather than arbitrary numerical groups.

Although all flights share the same origin and destination pair, their
continental-scale paths differ sufficiently for several recurring trajectory
families to emerge.

The four groups should therefore be interpreted as **geometric route
families** within the observed dataset.

No operational cause is assigned to a cluster solely from its geometric
location.

For example, the clustering result alone does not establish that a particular
route family was caused by an airline preference, weather condition, air
traffic control decision, or seasonal effect.

---

### 5.3 Representative Route Medoids

One representative observed trajectory was selected from each cluster using
the medoid definition introduced in Section 4.

The selected medoids are:

| Cluster | Members | Representative Flight |
|---|---:|---|
| Cluster 0 | 73 | `20260116T220252Z_adde95_JBU1124` |
| Cluster 1 | 558 | `20260111T074405Z_adabfe_JBU324` |
| Cluster 2 | 88 | `20260112T064839Z_a33328_AAL10` |
| Cluster 3 | 340 | `20260131T200523Z_a03765_AAL32` |

The representative routes are shown below.

![Representative route medoids](docs/figures/final_representative_routes_map.png)

*Figure 2. Medoid trajectory selected for each of the four final HDBSCAN
clusters.*

Because each medoid corresponds to a real observed trajectory, these paths can
be inspected directly as examples of the recurring route families identified
by the clustering process.

The mean distance from each medoid to the other trajectories in its cluster is:

| Cluster | Mean Distance to Medoid (km) | Median (km) | 95th Percentile (km) | Maximum (km) |
|---|---:|---:|---:|---:|
| Cluster 0 | 148.622 | 137.234 | 275.240 | 329.895 |
| Cluster 1 | 90.300 | 76.319 | 178.263 | 250.777 |
| Cluster 2 | 99.586 | 102.699 | 145.416 | 196.533 |
| Cluster 3 | 99.908 | 84.981 | 189.940 | 304.439 |

Cluster 0 has the largest mean distance from its medoid, suggesting greater
geometric spread within that route family.

Cluster 1, despite being the largest cluster, has the smallest mean distance
from its medoid among the four groups.

This indicates that cluster size alone does not determine geometric
compactness.

---

### 5.4 Separation Between Representative Routes

Pairwise Hausdorff distances between the four medoid trajectories provide a
direct measure of separation among the representative route families.

The medoid-distance matrix is:

| | Cluster 0 | Cluster 1 | Cluster 2 | Cluster 3 |
|---|---:|---:|---:|---:|
| **Cluster 0** | 0.000 | 634.983 | 505.640 | 451.906 |
| **Cluster 1** | 634.983 | 0.000 | 272.037 | 420.595 |
| **Cluster 2** | 505.640 | 272.037 | 0.000 | 213.922 |
| **Cluster 3** | 451.906 | 420.595 | 213.922 | 0.000 |

All values are in kilometres.

![Pairwise distances between representative route medoids](docs/figures/final_medoid_distances.png)

*Figure 3. Pairwise symmetric Hausdorff distances between the four
representative cluster medoids.*

The closest representative pair is:

```text
Cluster 2 ↔ Cluster 3
213.922 km
```

while the most separated pair is:

```text
Cluster 0 ↔ Cluster 1
634.983 km
```

The results indicate that even the two most similar representative route
families remain separated by more than 200 km under the Hausdorff measure.

---

### 5.5 Cluster Cohesion and Separation

Silhouette analysis was used as a complementary evaluation of the final
partition [9].

Because HDBSCAN noise trajectories are not members of a cluster, silhouette
values were calculated only for the:

```text
1,059 clustered trajectories
```

The overall silhouette score is:

```text
0.490665
```

Cluster-level results are:

| Cluster | Flights | Mean Silhouette | Median Silhouette | Minimum |
|---|---:|---:|---:|---:|
| Cluster 0 | 73 | 0.561794 | 0.594533 | 0.058089 |
| Cluster 1 | 558 | 0.541886 | 0.617965 | -0.264721 |
| Cluster 2 | 88 | 0.388803 | 0.396062 | 0.059164 |
| Cluster 3 | 340 | 0.417695 | 0.480134 | -0.234160 |

The silhouette distributions are shown below.

![Silhouette distribution by cluster](docs/figures/final_silhouette_distribution.png)

*Figure 4. Silhouette-value distribution for trajectories assigned to the four
final HDBSCAN clusters.*

Most trajectories have positive silhouette values.

Only:

```text
29 of 1,059 clustered trajectories
```

have negative silhouette values.

Specifically:

```text
Cluster 0:  0 negative trajectories
Cluster 1: 12 negative trajectories
Cluster 2:  0 negative trajectories
Cluster 3: 17 negative trajectories
```

These negative values occur near boundaries between route structures and do
not imply that the complete clustering is invalid.

Silhouette analysis was used only as complementary evaluation evidence.

It was **not** used to tune HDBSCAN parameters or select the final clustering
configuration.

This distinction is important because HDBSCAN identifies clusters from density
structure rather than by directly maximizing silhouette score.

---

### 5.6 January and July Composition

Month labels were not provided to HDBSCAN, but they can be examined after
clustering to understand the composition of the discovered groups.

The resulting month-by-cluster table is:

| Group | January | July | Total |
|---|---:|---:|---:|
| Cluster 0 | 53 | 20 | 73 |
| Cluster 1 | 142 | 416 | 558 |
| Cluster 2 | 49 | 39 | 88 |
| Cluster 3 | 221 | 119 | 340 |
| Noise | 89 | 206 | 295 |
| **Total** | **554** | **800** | **1,354** |

The composition is visualized below.

![January and July composition of final HDBSCAN groups](docs/figures/final_month_composition.png)

*Figure 5. January and July composition of the four final clusters and the
HDBSCAN noise set.*

The cluster compositions differ substantially.

For example, Cluster 1 contains considerably more July trajectories, whereas
Cluster 3 contains more January trajectories.

Noise membership also differs between the two months:

```text
January:
89 / 554 = 16.06%

July:
206 / 800 = 25.75%
```

These differences are **descriptive observations only**.

The study contains two selected months from a single year, and the January and
July source data have different provenance.

Therefore, the observed composition cannot be interpreted as evidence that
one season generally produces more route variability or anomalies than another.

---

### 5.7 Within-Cluster Trajectory Variation

In addition to medoid-based distances, pairwise distances among all
trajectories within each cluster were examined.

| Cluster | Flights | Mean Pairwise Distance (km) | Median (km) | 95th Percentile (km) |
|---|---:|---:|---:|---:|
| Cluster 0 | 73 | 196.696 | 180.349 | 357.418 |
| Cluster 1 | 558 | 122.793 | 113.129 | 234.296 |
| Cluster 2 | 88 | 131.468 | 123.089 | 262.249 |
| Cluster 3 | 340 | 137.415 | 123.110 | 267.101 |

These statistics provide a different interpretation from medoid distance.

The medoid statistics measure how far cluster members lie from one central
representative trajectory.

The pairwise statistics instead describe the total internal geometric
dispersion among all trajectories in a cluster.

Cluster 0 again exhibits the largest internal spread, while Cluster 1 shows the
smallest mean pairwise distance despite containing the largest number of
flights.

---

### 5.8 Primary Result Summary

The primary clustering analysis therefore identifies:

```text
4 recurring route families
1,059 clustered trajectories
295 HDBSCAN noise trajectories
4 observed representative medoid flights
```

The results provide evidence that multiple recurring horizontal route
structures exist within the LAX → JFK trajectory population.

However, the primary result alone does not establish whether these structures
are robust to methodological choices.

That question is addressed in Section 6 through:

```text
manual implementation validation
HDBSCAN parameter sensitivity
100-point vs 200-point representation sensitivity
Hausdorff vs discrete Fréchet sensitivity
```

The interpretation of the 295 noise trajectories is addressed separately in
Section 7.

## 6. Robustness and Sensitivity Analysis

### 6.1 Purpose of the Robustness Analysis

A clustering result can appear convincing visually while still depending
strongly on modelling choices.

For this reason, the final HDBSCAN partition was evaluated under several
independent changes to the methodology.

The robustness analysis addresses four questions:

1. Does the manual HDBSCAN implementation reproduce an established reference
   implementation?
2. Does the clustering remain similar when HDBSCAN parameters are changed?
3. Does increasing trajectory representation resolution alter the route
   families?
4. Does replacing Hausdorff distance with another trajectory metric change the
   scientific interpretation?

These tests distinguish **internal HDBSCAN stability** from broader
**methodological robustness**.

Internal HDBSCAN stability refers to persistence within the HDBSCAN hierarchy.

Methodological robustness refers to whether the final route structures remain
similar when reasonable modelling decisions are changed.

---

### 6.2 Manual HDBSCAN Validation Against scikit-learn

Because the primary clustering algorithm was implemented manually, an external
reference implementation was used to verify correctness.

The same precomputed Hausdorff-distance matrix and equivalent HDBSCAN settings
were supplied to `sklearn.cluster.HDBSCAN`.

An important convention difference was accounted for:

```text
Manual implementation:
min_samples = 15 nearest other trajectories

scikit-learn:
min_samples = 16 including the trajectory itself
```

After aligning this convention, the two implementations produced exactly the
same final partition.

| Validation Metric | Result |
|---|---:|
| Adjusted Rand Index, all trajectories | 1.000000 |
| Adjusted Rand Index, commonly clustered trajectories | 1.000000 |
| Noise-set Jaccard similarity | 1.000000 |
| Noise-mask agreement | 1.000000 |

The cluster/noise memberships of all 1,354 trajectories were identical.

This result provides strong evidence that the manual implementation correctly
reproduces the intended HDBSCAN clustering procedure.

The external implementation was used only as a validation reference; the
primary result remains generated by the manually implemented algorithm.

---

### 6.3 HDBSCAN Parameter Sensitivity

The final clustering configuration uses:

```text
min_samples = 15
min_cluster_size = 30
```

However, a single density-parameter choice should not be assumed to be uniquely
correct.

Nearby values of `min_samples` and `min_cluster_size` were therefore evaluated
to determine whether the main route families persisted.

Across reasonable neighbouring parameter configurations, the major route
structures remained recognizable.

The primary source of variation was not the disappearance of the major
clusters, but movement of trajectories near cluster boundaries between:

```text
cluster membership
        ↕
noise membership
```

This behaviour is expected in density-based clustering because trajectories
located in lower-density transition regions may change status when the density
requirement changes.

The sensitivity analysis also showed that some more aggressive parameter
combinations could produce abrupt hierarchy-selection changes.

Such configurations were treated as fragile rather than being preferred simply
because they produced a particular number of clusters or anomalies.

The final configuration was therefore selected as a stable and interpretable
operating point rather than as a parameter combination chosen to maximize
silhouette score or noise count.

---

### 6.4 Trajectory Representation Sensitivity

The primary trajectory representation contains:

```text
100 arc-length points per flight
```

To determine whether this geometric compression materially affects the
clustering, the complete analysis was repeated using:

```text
200 arc-length points per flight
```

This doubled the geometric resolution of every trajectory.

The full 200-point pairwise Hausdorff matrix contained the same:

$$
915{,}981
$$

unique trajectory pairs.

The 100-point and 200-point Hausdorff matrices were extremely similar.

| Distance-Matrix Comparison | Result |
|---|---:|
| Pearson correlation | 0.999986 |
| Median absolute difference | 0.459 km |
| 95th percentile absolute difference | 2.362 km |
| Maximum absolute difference | 18.527 km |

The full 200-point distance-matrix construction required approximately:

```text
1,288 seconds
≈ 21.5 minutes
```

demonstrating the additional computational cost of doubling the representation
resolution.

---

### 6.5 100-Point vs 200-Point Clustering

The final HDBSCAN result using the 100-point representation was:

```text
4 clusters
295 noise trajectories
```

The corresponding 200-point representation produced:

```text
4 clusters
291 noise trajectories
```

The sorted cluster sizes were:

```text
100 points:
73, 88, 340, 558

200 points:
73, 89, 344, 557
```

The two complete partitions were compared quantitatively.

| Robustness Metric | Result |
|---|---:|
| Adjusted Rand Index, all trajectories | 0.989996 |
| Adjusted Rand Index, commonly clustered trajectories | 1.000000 |
| Commonly clustered trajectories | 1,058 |
| Noise-set Jaccard similarity | 0.979730 |
| Noise-mask agreement | 0.995569 |

Only six trajectories changed cluster/noise status:

```text
100-point noise → 200-point cluster: 5 trajectories
100-point cluster → 200-point noise: 1 trajectory
```

Among trajectories assigned to clusters by both configurations, the cluster
partition was identical.

This is an important result.

It indicates that increasing the geometric representation from 100 to 200
points does not alter the four principal route families.

The observed sensitivity is concentrated almost entirely among a very small
number of trajectories close to the cluster/noise boundary.

Therefore, the 100-point representation was retained as the primary
configuration because it provides essentially the same scientific clustering
at substantially lower computational cost.

---

### 6.6 Trajectory-Distance Sensitivity

Hausdorff distance is the primary trajectory metric in this study.

However, because Hausdorff distance treats trajectories largely as geometric
point sets, the clustering was also repeated using **discrete Fréchet
distance**, which explicitly considers trajectory ordering.

This provides a stronger test than simply modifying an HDBSCAN parameter,
because the underlying definition of trajectory similarity itself is changed.

The complete discrete Fréchet matrix was calculated for all 1,354
trajectories.

Its construction required approximately:

```text
4,439 seconds
≈ 74 minutes
```

which was substantially more expensive than the Hausdorff calculation.

---

### 6.7 Hausdorff vs Discrete Fréchet Distance

Despite their different definitions, the two complete trajectory-distance
matrices were highly similar.

| Distance-Matrix Comparison | Result |
|---|---:|
| Pearson correlation | 0.999927 |
| Median absolute difference | 0.000 km |
| 95th percentile absolute difference | 0.633 km |
| Maximum absolute difference | 176.098 km |

Approximately 93.7% of all trajectory-pair distances were effectively equal
under the numerical comparison used in the analysis.

Discrete Fréchet distance was never smaller than the corresponding Hausdorff
distance in the calculated pairwise comparisons, consistent with the
relationship between the two measures for the discrete trajectory
representations considered here.

The resulting HDBSCAN partitions were:

```text
Hausdorff:
4 clusters
295 noise trajectories

Discrete Fréchet:
4 clusters
292 noise trajectories
```

The sorted cluster sizes were:

```text
Hausdorff:
73, 88, 340, 558

Discrete Fréchet:
77, 88, 340, 557
```

Partition agreement was extremely high.

| Robustness Metric | Result |
|---|---:|
| Adjusted Rand Index, all trajectories | 0.993846 |
| Adjusted Rand Index, commonly clustered trajectories | 1.000000 |
| Commonly clustered trajectories | 1,058 |
| Noise-set Jaccard similarity | 0.983108 |
| Noise-mask agreement | 0.996307 |

Only five trajectories changed cluster/noise status:

```text
Hausdorff noise → Fréchet cluster: 4 trajectories
Hausdorff cluster → Fréchet noise: 1 trajectory
```

Once again, trajectories assigned to clusters by both configurations had
identical cluster memberships.

This demonstrates that the four principal route families are not artifacts of
the Hausdorff metric.

Hausdorff distance was retained for the primary analysis because it generated
essentially the same scientific partition while requiring substantially less
computation.

---

### 6.8 Nearest-Neighbour Structure

The sensitivity experiments also compared local nearest-neighbour structure.

For the 100-point and 200-point Hausdorff representations:

```text
Exact nearest-neighbour agreement: 73.04%
Mean top-5 neighbour overlap:       83.74%
Mean top-10 neighbour overlap:      88.01%
```

For Hausdorff and discrete Fréchet distance:

```text
Exact nearest-neighbour agreement: 94.09%
Mean top-5 neighbour overlap:       95.58%
Mean top-10 neighbour overlap:      97.07%
```

These values show that local pairwise ordering can change slightly even when
the final density-based clustering remains almost unchanged.

This distinction is important.

A robustness analysis should not require every pairwise relationship to remain
identical.

Instead, the central question is whether reasonable changes in representation
or distance definition alter the higher-level scientific structure being
reported.

In this study, they do not.

---

### 6.9 Interpretation of the Robustness Results

The sensitivity analyses consistently lead to the same conclusion.

The four major trajectory families are highly stable.

Three independent comparisons support this:

```text
Manual HDBSCAN vs scikit-learn
ARI = 1.000000

100-point vs 200-point representation
ARI = 0.989996

Hausdorff vs discrete Fréchet distance
ARI = 0.993846
```

Even more importantly, for both major modelling sensitivity experiments:

```text
ARI among commonly clustered trajectories = 1.000000
```

Therefore, when a trajectory remains assigned to a cluster under both
configurations, its route-family assignment does not change.

The small differences observed across configurations occur primarily at the
boundary between cluster membership and HDBSCAN noise.

This means that the dominant scientific conclusion of the clustering analysis
is robust:

> Four recurring horizontal trajectory families are consistently present in
> the analysed LAX → JFK dataset, while modelling sensitivity is concentrated
> primarily among borderline noise/cluster cases.

This robustness is particularly important for the anomaly analysis.

It suggests that persistent noise trajectories can be distinguished from cases
whose anomaly status is sensitive to a single modelling choice.

The next section therefore investigates the HDBSCAN noise population in greater
detail.

## 7. Anomaly-Candidate Analysis

### 7.1 Purpose of the Anomaly Analysis

The primary HDBSCAN configuration identifies:

```text
295 noise trajectories
```

out of:

```text
1,354 total trajectories
```

corresponding to:

```text
21.79% of the dataset
```

These trajectories are not automatically interpreted as abnormal, unsafe, or
operationally problematic flights.

In this study, an HDBSCAN noise label means only that the trajectory was not
retained as a member of one of the four recurring density-based route families.

A noise trajectory may result from:

- a legitimate but uncommon routing;
- a temporary reroute;
- weather avoidance;
- air traffic control intervention;
- operational constraints;
- diversion-related behaviour;
- unusual but valid flight planning;
- incomplete trajectory coverage; or
- data-quality problems.

For this reason, the analysis uses the term **anomaly candidate** rather than
confirmed anomaly.

The anomaly investigation is conducted after the primary clustering has already
been completed.

No weather data, event dates, or known disruption labels were used as inputs to
HDBSCAN.

---

### 7.2 Persistent Noise Across HDBSCAN Configurations

A trajectory labelled as noise under only one parameter configuration may be a
borderline case rather than a strongly unusual route.

To evaluate this, noise membership was compared across three nearby HDBSCAN
configurations using different `min_samples` settings while maintaining the same
general clustering methodology.

Among the 295 noise trajectories in the primary configuration:

| Persistence | Number of Trajectories |
|---|---:|
| Noise in 1 of 3 configurations | 0 |
| Noise in 2 of 3 configurations | 56 |
| Noise in all 3 configurations | 239 |
| **Total primary noise** | **295** |

Therefore:

```text
239 / 295 = approximately 81.0%
```

of the primary noise trajectories remained noise in all three tested
configurations.

This persistence provides additional evidence that most of the primary noise
population is not simply produced by one specific `min_samples` value.

However, persistence still does not determine the physical cause of the
trajectory deviation.

---

### 7.3 Distance to the Nearest Representative Route

To measure geometric extremeness, each primary noise trajectory was compared
with the four representative cluster medoids.

For each noise trajectory, the minimum Hausdorff distance to any cluster medoid
was recorded.

The resulting nearest-medoid distances were:

| Statistic | Distance (km) |
|---|---:|
| Minimum | 92.486 |
| Median | 201.418 |
| 95th percentile | 443.453 |
| Maximum | 799.939 |

The 95th-percentile value:

```text
443.453 km
```

was used as a descriptive threshold for identifying the most geometrically
extreme primary noise trajectories.

This threshold does not redefine the HDBSCAN result.

It is used only after clustering to help prioritize noise trajectories for
further investigation.

---

### 7.4 Core-Distance Characteristics of Noise

The HDBSCAN core distances of the primary noise trajectories were also
examined.

The distribution was:

| Statistic | Core Distance (km) |
|---|---:|
| Minimum | 70.379 |
| Median | 115.341 |
| 95th percentile | 333.332 |
| Maximum | 654.729 |

Large core distances indicate that a trajectory lies in a relatively sparse
region of trajectory space.

However, core distance and nearest-medoid distance describe different
properties.

Core distance measures local trajectory-space density, whereas nearest-medoid
distance measures geometric separation from the nearest representative route.

For this reason, neither quantity is treated as a complete anomaly score.

---

### 7.5 Ranking of Anomaly Candidates

The primary noise trajectories were ranked using a transparent
post-clustering procedure.

The ranking priority was:

```text
1. persistence across HDBSCAN configurations
2. nearest-medoid distance
3. core distance
```

Persistence was ranked first because a trajectory remaining noise across nearby
parameter configurations provides stronger evidence of robust geometric
non-conformance.

Nearest-medoid distance was then used to prioritize trajectories that lie far
from all four recurring representative routes.

Core distance was used as an additional indication of local isolation.

No arbitrary weighted anomaly score was introduced.

This preserves direct interpretability of the ranking criteria.

---

### 7.6 Visual Inspection of the Highest-Ranked Candidates

The twelve highest-ranked anomaly candidates were examined individually.

Their trajectories showed substantial geometric differences from the dominant
LAX → JFK route structures.

A notable pattern emerged during this inspection:

```text
8 of the top 12 candidates occurred on July 28, 2026
```

The highest-ranked candidate was:

```text
20260728T184937Z_a5124c_DAL939
```

Other highly ranked July 28 trajectories included flights corresponding to
callsigns:

```text
DAL958
DAL979
AAL118
AAL2
DAL934
AAL274
DAL915
```

The concentration of several independently detected anomaly candidates on the
same date motivated a subsequent date-level analysis.

Importantly, July 28 was not selected before clustering.

The date became interesting only after examining the unsupervised anomaly
results.

---

### 7.7 Date-Level Concentration of Anomaly Candidates

The anomaly population was grouped by date to determine whether unusual
trajectories occurred randomly through time or were concentrated on particular
days.

A particularly strong signal was observed around July 28, 2026.

The three-day period from July 27 to July 29 produced:

| Date | Total Flights | HDBSCAN Noise | Persistent Noise | Geometrically Extreme |
|---|---:|---:|---:|---:|
| July 27 | 29 | 16 | 14 | 2 |
| July 28 | 20 | 14 | 14 | 8 |
| July 29 | 26 | 11 | 9 | 0 |

For July 28:

```text
14 / 20 = 70.0%
```

of the flights were labelled as HDBSCAN noise.

All 14 of those July 28 noise trajectories were persistent across the three
evaluated HDBSCAN configurations.

Furthermore:

```text
8 / 20 = 40.0%
```

of all July 28 flights exceeded the 95th-percentile nearest-medoid distance
threshold.

---

### 7.8 Concentration of the Most Geometrically Extreme Cases

Across the complete dataset, 15 primary noise trajectories exceeded the
95th-percentile nearest-medoid distance threshold of 443.453 km.

Of these 15 trajectories:

```text
8 occurred on July 28
```

Therefore:

```text
8 / 15 = approximately 53.3%
```

of the most geometrically extreme primary noise cases occurred on a single day.

At the same time, July 28 contains only:

```text
20 / 1,354 = approximately 1.48%
```

of all trajectories in the complete dataset.

This produces a strong temporal concentration of extreme geometric deviations.

The result does not identify the cause of the deviations, but it provides a
specific event window for targeted investigation.

---

### 7.9 Comparison with Adjacent Dates

The immediately surrounding dates provide useful descriptive context.

On July 27:

```text
Noise rate:            55.17%
Persistent-noise rate: 48.28%
Extreme cases:         2
```

On July 28:

```text
Noise rate:            70.00%
Persistent-noise rate: 70.00%
Extreme cases:         8
```

On July 29:

```text
Noise rate:            42.31%
Persistent-noise rate: 34.62%
Extreme cases:         0
```

The median nearest-medoid distance among noise trajectories also differed
strongly:

```text
July 27: 245.53 km
July 28: 599.08 km
July 29: 150.66 km
```

The maximum nearest-medoid distances were:

```text
July 27: 501.31 km
July 28: 799.94 km
July 29: 217.10 km
```

The July 28 signal therefore involves not only a larger proportion of noise
trajectories but also substantially greater geometric deviation among those
noise cases.

---

### 7.10 Month-Level Context

At the broader month level, the primary HDBSCAN result contains:

| Month | Total Flights | Noise | Noise Rate | Persistent Noise | Persistent Rate | Extreme Cases |
|---|---:|---:|---:|---:|---:|---:|
| January | 554 | 89 | 16.06% | 64 | 11.55% | 1 |
| July | 800 | 206 | 25.75% | 175 | 21.88% | 14 |

July contains a larger proportion of noise, persistent noise, and extreme
cases than January.

However, this result must not be interpreted as evidence of a general seasonal
effect.

Only two months from one year are included, and the source-data provenance of
January and July differs.

The month-level comparison is therefore descriptive only.

The event-level July 28 concentration is considered separately because it was
identified as a localized pattern within the unsupervised clustering results.

---

### 7.11 Data-Quality Diagnostics for Top Candidates

Large trajectory deviations can also be produced or exaggerated by missing
observations or temporal gaps.

The highest-ranked anomaly candidates were therefore subjected to additional
quality diagnostics.

Some trajectories contain relatively large observation gaps, including gaps on
the order of approximately:

```text
910 to 1,730 seconds
```

in the available reduced data.

Such gaps can increase uncertainty about the exact path flown between observed
positions.

The available July data are approximately 10-second reduced trajectories rather
than the original high-resolution ADS-B records.

Therefore, unusual route geometry observed across large gaps cannot be treated
as fully validated flight behaviour without examining the higher-resolution
source observations.

This is especially important for the July 28 candidates.

The clustering result identifies them as geometrically unusual based on the
available trajectory data, but high-resolution ADS-B validation remains
necessary before drawing stronger operational conclusions.

---

### 7.12 Current Interpretation of the July 28 Event

The strongest conclusion currently supported by the data is:

> The unsupervised clustering independently identified an unusually
> concentrated set of persistent and geometrically extreme trajectories on
> July 28, 2026. The cause has not yet been established.

This wording is intentionally conservative.

The current analysis supports the existence of a geometric event signal.

It does not yet establish whether the observed deviations were produced by:

```text
convective weather
air traffic control rerouting
airspace restrictions
traffic-flow management
airline operational decisions
data-quality limitations
or another factor
```

Determining the cause requires independent evidence.

---

### 7.13 Planned Event-Level Validation

The next anomaly-investigation stage will focus on July 27–29, with July 27
and July 29 providing adjacent-date comparison cases.

The planned validation sequence is:

```text
unsupervised anomaly detection
        ↓
identify July 28 concentration
        ↓
retrieve higher-resolution ADS-B trajectories
        ↓
validate trajectory continuity and geometry
        ↓
obtain independent weather / operational information
        ↓
compare route deviations with external event evidence
        ↓
interpret likely operational cause
```

The higher-resolution ADS-B data will first be used to determine whether the
observed route deviations remain present when temporal gaps and reduced
sampling are examined more closely.

Only after this validation should external meteorological or operational
evidence be used to interpret the event.

This ordering is important because it prevents the clustering model from being
constructed around a known weather event.

Instead, the event is discovered from trajectory geometry first and explained
only afterward.

---

### 7.14 Status of the Anomaly Investigation

The trajectory-clustering methodology and primary route-pattern analysis are
complete and validated.

The anomaly-investigation stage is still active and is a central part of the
research.

The current study has established:

```text
295 primary anomaly candidates

239 persistent noise trajectories

15 geometrically extreme primary noise trajectories

8 of the 15 extreme cases on July 28

14 of 20 July 28 flights labelled persistent noise
```

The unresolved question is no longer whether July 28 contains an unusual
trajectory pattern.

The unresolved question is **why that pattern occurred and whether it reflects
a genuine operational event after higher-resolution validation**.

Until that investigation is completed, no causal explanation is assigned.

## 8. Discussion

### 8.1 Principal Findings

The analysis identifies four recurring horizontal trajectory families among
1,354 flights operating from LAX to JFK.

Although all flights share the same origin and destination, their observed
continental-scale paths do not form a single homogeneous route population.
Instead, the HDBSCAN analysis reveals several recurring geometric structures,
together with a substantial population of trajectories that do not belong
strongly to those structures.

The primary clustering contains:

```text
4 recurring route families
1,059 clustered trajectories
295 HDBSCAN noise trajectories
```

The significance of this result is not simply that four clusters were obtained.

More important is that the four-cluster structure remains almost unchanged
when several major modelling choices are altered.

The principal route families therefore appear to represent persistent
geometric structure in the analysed trajectory population rather than an
artifact of one specific implementation or parameter choice.

---

### 8.2 Robustness of the Route Families

One of the central findings of this study is the strong robustness of the
identified trajectory groups.

The manual HDBSCAN implementation reproduces the equivalent scikit-learn
partition exactly:

```text
ARI = 1.000000
```

When the trajectory representation is increased from 100 to 200 arc-length
points, the resulting partition remains extremely similar:

```text
ARI = 0.989996
```

When the trajectory-distance definition is changed from symmetric Hausdorff
distance to discrete Fréchet distance, agreement remains similarly high:

```text
ARI = 0.993846
```

In both major modelling-sensitivity experiments, trajectories assigned to
clusters under both configurations preserve exactly the same route-family
partition:

```text
ARI among commonly clustered trajectories = 1.000000
```

This result provides an important distinction between two types of sensitivity.

The **major route structure is highly stable**, whereas a small number of
trajectories close to low-density boundaries can move between cluster
membership and noise.

Such behaviour is reasonable for density-based clustering and suggests that the
uncertainty of the model is concentrated primarily around borderline
trajectories rather than around the existence or identity of the four dominant
route families.

---

### 8.3 Implications of the Trajectory Representation

The trajectory representation used in this study separates temporal sampling
from geometric comparison.

The approximately 10-second trajectories provide a harmonised working dataset,
while the 100-point arc-length representation describes each complete flight as
a fixed geometric path.

This distinction is useful because flights with different durations and
numbers of ADS-B observations can still be compared according to their spatial
route geometry.

The 100-versus-200-point sensitivity experiment provides direct evidence that
100 points are sufficient for the primary scientific objective of this study.

Doubling the representation resolution produced only small changes in the
pairwise distance matrix and almost no change in the resulting clustering.

Therefore, increasing trajectory resolution beyond 100 points would increase
computational cost substantially without materially changing the identified
route families for this dataset.

This conclusion should not be interpreted as a universal claim that 100 points
are sufficient for every trajectory-clustering problem.

The appropriate representation resolution depends on route scale, trajectory
complexity, and the scientific objective.

---

### 8.4 Implications of the Distance-Metric Comparison

Symmetric Hausdorff distance was selected as the primary whole-trajectory
metric because it provides an interpretable measure of geometric route
separation and has previously been used in aviation trajectory clustering
[2,3].

However, Hausdorff distance does not explicitly model traversal ordering in the
same way as discrete Fréchet distance.

For this reason, a complete Fréchet-based clustering experiment was performed
rather than assuming that the Hausdorff result was uniquely valid.

The extremely high agreement between the two final partitions indicates that
the dominant route structures in this dataset are sufficiently geographically
distinct that both distance formulations recover essentially the same
scientific grouping.

This strengthens the interpretation of the clusters.

The result is not dependent on a single mathematical definition of trajectory
similarity.

Hausdorff distance was therefore retained not because it was the only metric
capable of identifying the route families, but because it produced essentially
the same clustering at substantially lower computational cost.

---

### 8.5 Relationship to Previous Flight-Trajectory Clustering Research

The methodological foundation of this project is closely related to previous
aviation trajectory-clustering research.

Kim and Mavris [2] used Hausdorff trajectory distance together with HDBSCAN to
group historical flight trajectories, select representative real-flight cases,
and examine irregular trajectories represented by HDBSCAN noise.

The present study therefore does **not** claim that the combination of
Hausdorff distance and HDBSCAN is itself a novel contribution.

Instead, the present work builds upon this established methodology and extends
the analysis in several directions.

First, the preprocessing stage explicitly investigates stale positions,
spatial discontinuities, trajectory fragmentation, and the consequences of
different source-data resolutions.

Second, the geometric representation is evaluated systematically rather than
being treated as a fixed preprocessing decision.

Third, the trajectory-distance choice is tested through a complete
Hausdorff-versus-Fréchet sensitivity experiment.

Fourth, the complete HDBSCAN procedure is implemented manually and independently
validated against scikit-learn.

Fifth, representative flights are selected as medoids directly in the same
Hausdorff distance space used for clustering.

Finally, HDBSCAN noise is investigated through persistence, geometric
extremeness, and temporal concentration rather than being interpreted directly
as confirmed operational anomaly.

These extensions shift the emphasis from simply obtaining trajectory clusters
toward determining whether the identified structures and anomaly candidates
remain credible under alternative modelling assumptions.

---

### 8.6 Representative Routes and Operational Interpretation

The medoid-based representative trajectories provide real observed examples of
each recurring route family.

This is preferable for interpretation to generating an artificial average path
because every representative can be traced back to an actual flight.

The four medoids are separated geographically, with pairwise Hausdorff
distances ranging from:

```text
213.922 km
```

to:

```text
634.983 km
```

The separation supports the interpretation that the clusters represent
different large-scale route structures.

However, trajectory geometry alone does not explain why those structures
exist.

Possible influences could include:

- air traffic flow organization;
- route availability;
- winds;
- weather avoidance;
- airline flight planning;
- air traffic control constraints;
- departure timing; and
- other operational factors.

Determining the relative importance of these factors is outside the information
contained in the primary horizontal trajectory-distance matrix.

Therefore, the clusters should be interpreted first as **empirically observed
geometric route families**, not as predefined operational route categories.

---

### 8.7 Interpretation of HDBSCAN Noise

The noise population requires more careful interpretation than the main route
clusters.

HDBSCAN identifies observations that are not retained as part of sufficiently
persistent dense structures.

This is a geometric and density-based statement.

It is not equivalent to saying that a flight was unsafe, erroneous, or
operationally abnormal.

This distinction is consistent with previous trajectory-anomaly research,
where unusual trajectory behaviour is detected first and contextual
information is subsequently used to investigate its cause [5,6].

The persistence analysis strengthens the anomaly investigation because:

```text
239 of 295 primary noise trajectories
```

remain noise across all three neighbouring HDBSCAN configurations.

Therefore, most of the primary noise population cannot be explained simply as
a consequence of one particular `min_samples` value.

Nevertheless, persistent noise may still correspond to valid uncommon routes.

External operational evidence remains necessary for causal interpretation.

---

### 8.8 Significance of the July 28 Concentration

The most notable anomaly-related finding is the concentration of persistent and
geometrically extreme trajectories on July 28, 2026.

The date was not selected as an event beforehand.

Instead, the sequence was:

```text
trajectory clustering
        ↓
noise identification
        ↓
persistence analysis
        ↓
geometric-extremeness analysis
        ↓
date-level aggregation
        ↓
July 28 concentration discovered
```

This order is scientifically important because it reduces the risk of
constructing the analysis around a known event.

July 28 contains only:

```text
20 / 1,354 ≈ 1.48%
```

of the complete dataset but contains:

```text
8 / 15 ≈ 53.3%
```

of the geometrically extreme primary noise cases.

Furthermore, all 14 trajectories classified as primary noise on that date
remain noise across the tested neighbouring HDBSCAN configurations.

The July 28 noise trajectories also have substantially larger nearest-medoid
distances than the corresponding noise populations on July 27 and July 29.

Together, these observations indicate that July 28 contains a genuine
**trajectory-space concentration requiring explanation**.

They do not yet establish the explanation itself.

---

### 8.9 Detection Versus Causal Explanation

A central principle of this study is the separation between:

```text
detecting an unusual trajectory pattern
```

and:

```text
explaining why that pattern occurred
```

The current clustering results support the first statement.

They do not yet support the second.

Potential causes of the detected route deviations may include meteorological
conditions, traffic-flow management, airspace restrictions, air traffic
control actions, airline operational decisions, or combinations of these
factors.

Data-quality limitations must also be considered because the available July
trajectories were derived from reduced approximately 10-second data.

Some high-ranked candidates contain substantial observation gaps, meaning that
the exact route between recorded positions cannot always be reconstructed from
the currently available dataset.

Consequently, the next research stage requires two independent forms of
validation:

1. higher-resolution trajectory validation; and
2. external environmental or operational evidence.

Only when these sources are examined together should a physical explanation be
proposed.

---

### 8.10 January and July Differences

The final clustering contains different proportions of January and July
trajectories within the four route families and noise population.

In particular:

```text
January noise rate = 16.06%
July noise rate    = 25.75%
```

July also contains more persistent and geometrically extreme noise
trajectories.

These observations may motivate further investigation, but they are not
sufficient to establish a seasonal relationship.

There are several reasons for caution.

Only January and July of one year are included.

The two months also have different source-data provenance, with higher-resolution
January source observations available but only reduced July source data
available for the primary analysis.

Furthermore, many operational factors other than season can influence
continental flight routing.

The month comparison is therefore treated as descriptive metadata rather than
as evidence that summer operations are intrinsically more anomalous than winter
operations.

A seasonal research question would require additional months, preferably across
multiple years and with consistent source-data quality.

---

### 8.11 Methodological Strengths

Several characteristics strengthen the credibility of the current results.

The complete clustering object is clearly defined as one flight trajectory
rather than individual ADS-B observations.

Data-quality processing is separated from anomaly detection.

The trajectory representation is tested at multiple resolutions.

The primary trajectory distance is compared with an alternative metric.

The manual HDBSCAN implementation is verified independently.

Parameter sensitivity is examined rather than reporting only a single
configuration.

Noise persistence is distinguished from HDBSCAN's internal cluster stability.

Finally, anomaly candidates are not assigned an operational explanation without
independent evidence.

Taken together, these choices make the study more reproducible and reduce the
dependence of its conclusions on hidden modelling assumptions.

---

### 8.12 Limitations

Despite the robustness of the clustering result, several limitations remain.

#### Single Origin-Destination Pair

The analysis considers only LAX → JFK flights.

The four identified route families therefore describe this specific
origin-destination population and should not automatically be generalized to
other markets.

#### Limited Temporal Coverage

Only January and July 2026 are included.

The dataset cannot establish long-term seasonal patterns or year-to-year
variability.

#### Unequal Source-Data Provenance

January and July were not available at identical original temporal resolution.

This particularly limits direct month-level comparisons and detailed validation
of July anomaly candidates.

#### Horizontal Geometry Only

The primary trajectory distance uses latitude and longitude.

Altitude, vertical rate, speed, aircraft type, winds, weather, and operational
constraints are not included directly in the clustering model.

Two flights following similar horizontal paths but substantially different
vertical profiles would therefore appear similar under the primary metric.

#### Hausdorff Sensitivity to Local Deviations

Hausdorff distance emphasizes the most separated portion of two trajectories.

This makes it useful for identifying substantial route deviations but also
means that isolated geometric irregularities or data problems can increase the
distance strongly.

The preprocessing and Fréchet sensitivity analysis reduce this concern but do
not eliminate it completely.

#### Unvalidated Event Cause

The July 28 concentration is currently a detected geometric event rather than a
causally explained operational event.

Higher-resolution ADS-B and independent contextual evidence remain necessary.

---

### 8.13 Scientific Contribution of the Present Study

The scientific contribution of the project is therefore best characterized not
as the invention of a new clustering algorithm, but as the development and
validation of a **robust trajectory-clustering and anomaly-investigation
framework**.

The framework combines:

```text
ADS-B quality validation
        ↓
geometric trajectory harmonisation
        ↓
trajectory-level distance modelling
        ↓
manual density-based clustering
        ↓
independent implementation validation
        ↓
representation and metric sensitivity
        ↓
representative-route identification
        ↓
persistent anomaly-candidate detection
        ↓
event-level investigation
```

The strongest completed result is that the four recurring LAX → JFK route
families remain highly stable under substantial methodological changes.

The most promising ongoing result is the independently discovered
concentration of persistent and geometrically extreme trajectories identified
in the event-level analysis.

If subsequent high-resolution and external-event validation confirms that this
concentration corresponds to a documented operational or meteorological event,
the anomaly-investigation component would provide an important extension of the
current study.

At the present stage, however, that interpretation remains an active research
question rather than a completed conclusion.

## 9. Future Work

The current study establishes a robust trajectory-clustering framework and
identifies a promising concentration of persistent geometric anomaly candidates.

However, several extensions are required before stronger operational or
causal conclusions can be made.

### 9.1 High-Resolution Event Validation

The high-resolution data should therefore be used to verify:

- whether the large route deviations remain visible;
- whether apparently extreme trajectory segments are supported by continuous
  observations;
- whether reduced-data gaps distort the inferred path;
- whether position freshness and continuity are acceptable; and
- whether the anomaly candidates represent real aircraft motion rather than
  data-quality artifacts.

July 27 and July 29 should be retained as adjacent-date comparison cases rather
than analysing July 28 in isolation.

---

### 9.2 Independent Weather and Operational Validation

After the trajectory geometry has been validated, the next step is to compare
the detected deviations with independent contextual information.

Potential evidence includes:

- convective-weather observations;
- radar-derived precipitation or storm products;
- significant meteorological advisories;
- winds and upper-air conditions;
- temporary airspace restrictions;
- traffic-flow management initiatives;
- air traffic control rerouting information; and
- airport or airline operational disruption records.

The purpose of this stage is not to search for any event that appears to match
the clustering result.

Instead, independent evidence should be used to test specific hypotheses about
why the July 28 trajectories deviated from the recurring route families.

If a spatial and temporal correspondence is found, the study may conclude that
the anomalous trajectory concentration **coincided with** the documented
operational or meteorological condition.

A stronger causal statement should be made only if the available evidence
supports it.

---

### 9.3 Event-Level Spatial Comparison

If suitable weather or operational data become available, the anomaly
investigation can be extended geographically.

For example, the following information could be displayed together:

```text
representative route medoids
        +
July 27 trajectories
        +
July 28 trajectories
        +
July 29 trajectories
        +
weather or operational regions
```

This would allow the analysis to determine whether aircraft deviated around
specific geographic regions and whether those deviations were systematically
related to an external event.

Such an analysis would provide stronger evidence than comparing only daily
anomaly counts.

---

### 9.4 Additional Time Periods

The current study uses January and July 2026.

Additional months should be included to determine whether the discovered route
families persist across a broader range of operating conditions.

A stronger temporal study could include:

- multiple months from the same year;
- all four seasons;
- the same months across multiple years; or
- targeted periods containing known major operational disruptions.

Consistent source-data resolution across these periods would be particularly
important.

Expanding the temporal coverage would also allow the current descriptive
January-versus-July comparison to be replaced by a more defensible analysis of
seasonal or operational variability.

---

### 9.5 Additional Origin-Destination Pairs

The current results apply only to LAX → JFK.

To determine whether the methodology generalizes, the complete workflow should
be applied to additional origin-destination pairs.

Potential comparisons could include:

- another long-haul transcontinental route;
- a shorter domestic route;
- a route with several common airway alternatives; and
- a route frequently affected by convective weather or airspace constraints.

This would help determine whether the strong robustness observed in the current
dataset is specific to LAX → JFK or is a more general property of the proposed
framework.

---

### 9.6 Inclusion of Vertical and Kinematic Information

The primary clustering in this study uses horizontal latitude-longitude
geometry only.

Future work could investigate whether additional trajectory dimensions improve
operational interpretation.

Candidate variables include:

```text
altitude
vertical rate
ground speed
heading
```

A three-dimensional or multivariate trajectory distance could potentially
distinguish flights that follow similar horizontal routes but use substantially
different vertical profiles.

However, additional variables should not simply be appended to the current
distance measure.

Their units, scaling, operational meaning, and data quality would need to be
considered explicitly.

The horizontal-only model should remain an interpretable baseline for such
extensions.

---

### 9.7 Comparison with Additional Clustering Baselines

The present study evaluates robustness across representation resolution,
trajectory-distance definition, HDBSCAN parameters, and an independent HDBSCAN
implementation.

A future publication-oriented study should also compare the final methodology
with alternative clustering approaches.

Possible baselines include:

```text
DBSCAN
hierarchical agglomerative clustering
k-medoids using the trajectory-distance matrix
```

The purpose would not necessarily be to identify a universally superior
algorithm.

Instead, the comparison would determine whether the main route families and
anomaly candidates are specific to HDBSCAN or are also supported by other
reasonable clustering formulations.

---

### 9.8 Development of a Continuous Anomaly Measure

The current anomaly framework deliberately avoids creating an arbitrary
weighted anomaly score.

Instead, anomaly candidates are described using interpretable quantities:

```text
HDBSCAN noise membership
persistence across parameter settings
nearest-medoid distance
core distance
```

Future work could investigate whether these quantities can be combined into a
well-calibrated continuous anomaly measure.

Such a measure would need to preserve interpretability and should be validated
against independently confirmed operational events.

Without such validation, a continuous anomaly score could provide numerical
precision without corresponding operational meaning.

---

### 9.9 Airline and Operational Metadata

Airline identity is currently retained as metadata but is not used as a
clustering feature.

Future analysis could investigate whether certain route families are more
frequently associated with particular airlines, departure periods, aircraft
types, or operational conditions.

Such relationships should be analysed after clustering rather than assuming
that airline identity explains route selection.

For example, an airline-dominated cluster may reflect scheduling, flight
planning, departure timing, network structure, or other confounding factors.

Therefore, any relationship between metadata and cluster membership should be
treated as association unless additional evidence supports a causal
interpretation.

---

### 9.10 Toward a Publishable Extended Study

The present work already establishes:

```text
a validated preprocessing pipeline
a complete manual HDBSCAN implementation
four robust route families
representative route medoids
multiple sensitivity analyses
persistent anomaly candidates
a concentrated event-level anomaly signal
```

To strengthen the work for publication, the most important additions are:

1. high-resolution validation of the July 28 trajectories;
2. independent weather or operational confirmation of the event;
3. application to additional origin-destination pairs or time periods; and
4. comparison with at least one reasonable alternative clustering baseline.

These extensions would move the contribution beyond demonstrating a robust
clustering pipeline toward showing that the framework can repeatedly identify
operationally meaningful trajectory anomalies.

The anomaly-investigation stage therefore remains the highest-priority next
research task.

## 10. Conclusion

This study developed and validated an unsupervised trajectory-clustering
framework for 1,354 LAX → JFK flights observed during January and July 2026.

Each complete flight was treated as one clustering sample. After data-quality
validation and temporal harmonisation, trajectories were represented using
fixed-length arc-length sampling and compared using symmetric Hausdorff distance
with Haversine point-to-point separation.

A manually implemented HDBSCAN algorithm identified four recurring route
families containing 1,059 trajectories, while 295 trajectories were labelled as
noise.

The four route families were represented by observed medoid trajectories,
providing real flight examples of the dominant geometric structures in the
dataset.

The clustering result was strongly supported by several independent validation
and sensitivity experiments.

The manual HDBSCAN implementation reproduced the equivalent scikit-learn
partition exactly:

```text
ARI = 1.000000
```

Doubling the geometric representation from 100 to 200 points produced:

```text
ARI = 0.989996
```

while replacing Hausdorff distance with discrete Fréchet distance produced:

```text
ARI = 0.993846
```

For both major modelling-sensitivity experiments, trajectories that remained
clustered under both configurations preserved exactly the same route-family
assignment.

These results indicate that the four major trajectory families are highly
robust and that methodological sensitivity is concentrated primarily near the
boundary between cluster membership and HDBSCAN noise.

The noise population was therefore investigated further as a source of anomaly
candidates.

Of the 295 primary noise trajectories, 239 remained noise across all three
tested neighbouring HDBSCAN configurations.

Post-clustering geometric analysis also identified 15 especially extreme noise
trajectories based on distance from the nearest representative route.

A notable temporal concentration emerged on July 28, 2026.

Although that date contains only approximately:

```text
1.48% of the complete dataset
```

it contains approximately:

```text
53.3% of the most geometrically extreme primary noise trajectories
```

and all 14 July 28 flights labelled as primary HDBSCAN noise remained noise
across the tested neighbouring parameter configurations.

This event was discovered from the unsupervised trajectory analysis rather than
being selected in advance.

However, the cause of the July 28 deviations has not yet been established.

The available July trajectories were derived from data reduced to approximately
10-second temporal resolution, and some high-ranked candidates contain
substantial observation gaps.

Therefore, higher-resolution ADS-B validation and independent meteorological or
operational evidence are required before attributing the detected deviations to
weather, air traffic control, airspace restrictions, or another cause.

The study therefore supports two main conclusions.

First, robust recurring horizontal route families can be identified from
historical ADS-B trajectories using trajectory-level distance measures and
density-based clustering.

Second, persistent HDBSCAN noise combined with geometric extremeness and
temporal concentration provides a useful framework for prioritizing unusual
trajectory events for further investigation.

The contribution of the present work is not the invention of the
Hausdorff–HDBSCAN combination itself, which has precedent in previous aviation
trajectory research.

Instead, the contribution lies in the rigorous validation of the complete
trajectory-clustering pipeline, the evaluation of sensitivity to representation
and distance choices, and the use of persistent geometric non-conformance as a
basis for event-level anomaly investigation.

The trajectory-clustering methodology and primary route-pattern analysis are
complete and validated.

The anomaly-investigation stage remains active and is a central part of the
research.

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
in *Advances in Knowledge Discovery and Data Mining*,
Lecture Notes in Computer Science, vol. 7819,
pp. 160–172, 2013.
doi: 10.1007/978-3-642-37456-2_14.

[8] R. J. G. B. Campello, D. Moulavi, A. Zimek, and J. Sander,
“Hierarchical density estimates for data clustering, visualization, and
outlier detection,”
ACM Transactions on Knowledge Discovery from Data,
vol. 10, no. 1, Article 5, pp. 5:1–5:51, 2015.
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
Christian Doppler Laboratory for Expert Systems,
Technical University of Vienna, Austria, 1994.