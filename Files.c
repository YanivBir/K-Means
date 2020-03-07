#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include "main.h"
#include "Point.h"

//This function read data from file (by words)
Point* readWordsFromFile(Point* arr, long *n, int *k, double *t, double *dt, int *limit, double *qm) {
	FILE * fp;
	fp = fopen(INPUT_FILE_PATH, "r");
	if (fp == NULL) {
		printf("Cannot open input file\n");
		MPI_Abort(MPI_COMM_WORLD, 1);
	}
	fscanf(fp, "%d %d %lf %lf %d %lf", n, k, t, dt, limit, qm);

	arr = (Point *)malloc((*n) * sizeof(Point));
	if (arr == NULL) {
		printf("Can't malloc arr in readWordsFromFile()\n");
		MPI_Abort(MPI_COMM_WORLD, 0);
	}

	for (long i = 0; i < *n; i++) {
		fscanf(fp, "%lf %lf %lf %lf %lf %lf", &(arr[i].x), &(arr[i].y), &(arr[i].z), &(arr[i].vX), &(arr[i].vY), &(arr[i].vZ));
	}
	fclose(fp);
	return arr;
}

//This function write the output data to the file
void writeDataToFile(char *data) {
	FILE * fp;
	fp = fopen(OUTPUT_FILE_PATH, "w");
	if (fp == NULL) {
		printf("Cannot create or open output file\n");
		MPI_Abort(MPI_COMM_WORLD, 1);
	}

	fprintf(fp, data);
	fclose(fp);
}