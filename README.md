# PaCMAP Operator

This operator performs PaCMAP (Pairwise Controlled Manifold Approximation) dimensionality reduction on image data in Tercen.

## Overview

PaCMAP is a dimensionality reduction algorithm that preserves both local and global structure of the data. It is particularly useful for visualization and exploration of high-dimensional data. This operator takes image data in a long format and applies PaCMAP to reduce it to a lower-dimensional representation.

## Input Data

The operator expects data in the following format:

### Main Projection
- `.y`: The pixel value
- `.ci`: Column index linking to the Column Projection
- `.ri`: Row index linking to the Row Projection

### Column Projection
- `eventId`: Unique identifier for each image/event
- `.ci`: Column index linking to the Main Projection

### Row Projection
- `pixel_id`: Unique identifier for each pixel position
- `.ri`: Row index linking to the Main Projection

Example of input data:

**Main Projection**
```
.ci .ri .y  
0   0  1.3  
1   0  0.1 
0   1  0.5 
1   1  0.8 
```

**Row Projection**
```
.ri pixel_id
0  1
1  2
```

**Column Projection**
```
.ci eventId
0  1
1  2
```

## Output Data

The operator outputs a data frame with the following columns:
- `PaCMAP_1`, `PaCMAP_2`, ...: The coordinates in the reduced dimensionality space
- `eventId`: The original event identifier
- `.ci`: Column index
- `.ri`: Row index (set to 0)

## Parameters

- `n_components` (default: 2): Number of dimensions in the embedding
- `n_neighbors` (default: 10): Number of neighbors for the kNN graph
- `MN_ratio` (default: 0.5): Ratio of mid-near pairs to be sampled
- `FP_ratio` (default: 2.0): Ratio of further pairs to be sampled

## References

- [PaCMAP GitHub Repository](https://github.com/YingfanWang/PaCMAP)
- Wang, Y., Huang, H., Rudin, C., & Shaposhnik, Y. (2021). Understanding How Dimension Reduction Tools Work: An Empirical Approach to Deciphering t-SNE, UMAP, TriMAP, and PaCMAP for Data Visualization. Journal of Machine Learning Research, 22(1), 3794-3841.