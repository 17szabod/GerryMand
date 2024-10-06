# Uniform Sampling of Redistricting Plans
This is an implementation of the algorithm described in this [preprint](https://arxiv.org/abs/2402.13868 "preprint")

## Installation
The package is written in Python, but geographic imaging software such as ArcGIS is recommended for handling and viewing the results.

### Required Packages
The following packages are required:
- networkx
- matplotlib
- numpy
- geopandas
- pandas
- sqlite3
### Recommended Packages
The isect_segments_bentley-ottman package available on [github](https://github.com/ideasman42/isect_segments-bentley_ottmann/ "github") is recommended for finding errors in the input data that may prevent the adjacency graph from being planar. If you opt not to use it, be sure to comment out any imports and usages.

## Execution Instructions
The recommended branch to use for windows is RemoteOnGerry. The master branch is intended for unix based distributions, but there aren't many differences between the two, and RemoteOnGerry has some QOL improvements, such as documentation. The primary executable is non_int_bottom_up.py. There is currently no UI, so parameters must be set in the code of non_int_bottom_up.py, at the end of the file. These parameters include the path to the shp file, the number of districts k, a discrete perimeter compactness parameter, and the number of samples to generate. They also include two uncomfortable parameters, the exit_edge and start_edge. These are the initial edges for determining the outer face of the embedding, and are used in generating the traversal. To find them, the current simplest way is to draw the graph (run with draw=true) and manually identify some edge on the outer face. If start_edge=exit_edge, the optimal exit_edge on the outer face will be used. The input is pruned and prepared in method enumerate_paths_with_order, which will likely need editing for each application.

## Contact
For further information or feature requests, please contact the author at dszabo2 (at) wisc.edu.
