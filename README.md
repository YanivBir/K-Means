# K-Means
Parallel and Distributed implementation of K-Means algorithm

This project was written by me as a final course assignment.
In this project, I implement the k-means algorithm for finding K-clusters, in a time range and
stop when I find a smaller Q (Quailty) than a given QM (Quailty).

I solved the problem by parallel and distributed implementation using OMP, CUDA and MPI

## The development processes
Before starting to implement the project, I wrote a small Python program in order to have an
output file to work on it. 
The program random K centers as starting points. For each point the
program random some points and gives them, a velocity vector in a way that makes them
converge to the central point at the end of the time.

At the begging, I implemented the k-means algorithm on one computer in sequential. I used a
small file (100 points) to check the implementation. Then I run a larger file.

Next I started to implement the parallel code.

## The Parallel implementation
At first, I implemented all the parts that can be parallel using OMP.

After measured times, I understood that long calculations on a very wide range of numbers
are more efficient to be performed on CUDA.
In the two following sections I will explain which parts parallel in OMP and which on
CUDA.

I used MPI to connect two additional computers, so they could also participate in the
calculations. At the beginning of the program the Master reads the input.txt file and updates
the slaves about the data. After that the Master send a selected time to one of the slave. The
slave runs the K-means algorithm and at the end, updates the Master if the desired QM is
found.

![Figure 2](https://github.com/YanivBir/K-Means/blob/master/figures/figure1.png)

## Parallel implementation in OMP
I parallel the following parts using OMP
* **Group Points:** This section is responsible for passing all the points and determine for
each point which cluster the point belongs. Since there is no dependence between one
point and another, this part can be performed simultaneously. The complexity of this part
is: o(n * k). (In the code this function appears as: choosePointCluster in kmeans.c file).
* **Calculating the centers of a clusters:** This section is responsible for passing all the
points assigned to a particular cluster to calculate the average of the coordinates X, Y,
and Z to find the new cluster center. In this section I sum all the points separately
because there is no dependence between one point and the other. The complexity of this
part is: o(n). (In the code this function appears as: calcClusterCenter in kmeans.c file)
* **Calculates the number of points assigned to a particular cluster:** The function passes
all points and calculate how many points were assigned to each cluster. The complexity
of the function is o(n) (In the code this function appears as: getClusterMemberCount in
kmeans.c file)
* **Copy the array assign clusters:** This array is an int type and its function is to classify
each point to the appropriate cluster. The complexity of the function is o(n) (It appears in
the code as: copyAssignArr in kmeans.c file).
* **Check if a point has moved to another cluster:** This function is responsible for going
through all the points and see whether they have different cluster from the previous k-
means iteration. It is possible to go over several points in parallel so I parallel this
function. Complexity o(n). (It appears in the code as: countChanges in kmeans.c file).
* **Get all the points that have been classified on a particular custer:** In this function, I
pass all the points and return an array of points that have been classified into a specific
cluster. I can do this at the same time. Complexity o(n) (It appears in the code as:
getClusterPoints in kmeans.c file)
* **Copying the points array:** I can copy the points array in parallel. complexity is o(n). (It
appears in the code as: copyPointsArray in kmeans.c file)
* **Initialize the cluster centers array:** I can initiate the points array in parallel. In
complexity is o(n). (It appears in the code as: initClusterCenters in kmeans.c file).

## Parallel implementation in CUDA
* **Calculating coordinates of points at a specific time:** I can parallel it in complexity
of o(n). (It appears in the code as: cuda_setCurrentPosition in kmeansCuda.c file).
* **Calculates the distance of points from the center of a clusters:** Can be parallel in
complexity of o(n * k). (It appears in the code as: cuda_ calcPointsDistance in
kmeansCuda.c file).
* **Calculating the diameter of a specific cluster:** Calculated by finding the maximum
distance between two points in a cluster Can be parallel, in complexity of o(n ^ 2) (It
appears in the code as: cuda_ findDiameter in kmeansCuda.c file).

## Systems Architecture
![Systems Architecture](https://github.com/YanivBir/K-Means/blob/master/figures/SystemsArchitecture.png)
