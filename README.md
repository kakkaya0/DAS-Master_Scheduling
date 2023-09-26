# DAS_Master_Scheduling
This repository implements a Master Scheduling algorithm for Demand Adaprive Services, which was published by Crainic, Errico, Malucelli and Nonato [link to paper](https://link.springer.com/article/10.1007/s10479-010-0710-5). The algorithm can be used to determine the master schedule of a single DAS line,
that is, determining the time windows for the compulsory stops of the line.

# Structure
In the data directory, you find the [TSPLIB instances](http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/) and the [TSPGL2 instances](https://w1.cirrelt.ca/~errico/#Instances) aswell as additional information about compulsory stops, node cover, feasibility cover, and route edges in the route directory.

In the generators directory, you find how to generate the data from the mentioned data sources to be used for the algorithm.

# Running the Algorithm
To run the algorithm you should run the following in your IDE terminal: python generators\Segmentation.py [width_type] [instance] [speed] [number_of_samples] [dimension_of_samples] [standard_deviation] [epsilon] [time_window_width]. The following input arguments are valid:

| Argument | Input|
| ---------|----------|
| width_type | fixed,adjustable |
| instance | This represents the instance number, each instance in the data directory has an instance number, this can be found in \data\routes (e.g. Design_0-14-10-0.8-0.2_hybrid has the instance number 14) |
| speed | Set of whole numbers |
| number_of_samples | Set of whole numbers |
| dimension_of_samples | Set of whole numbers |
| standard_deviation | 0 - 100 |
| epsilon | 0 - 1  |
| time_window_width | Set of whole numbers |

Running the algorithm with the instance 22 with a 10 samples and 10 dimensions for each sample and with a speed argument of 25 and standard deviation value of 2, epsilon value of 0.05 and the time_window_width of 5 can be done via python generators\Segmentation.py adjustable 22 25 10 10 2 0.05 5
