#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <mpi.h>
#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <windows.h>
#include <iostream>
#include <time.h>
#include "Point.h"
#include "Kmeans.h"
#include "main.h"
#include "Files.h"

int finalizeCuda();

//Main process func
int kmeanMainResult(Point* points, long n, int k, double currentTime, int limit, double qm) {
	char *outputString = (char *)malloc(sizeof(char)* LINE_SIZE * (5 + k));
	if (outputString == NULL)
	{
		printf("Can't malloc outputString kmeanMainResult()\n");
		MPI_Abort(MPI_COMM_WORLD, 0);
	}
	sprintf(outputString, "%s", "");
	int kmeanRet = 0;
	kmeanRet = kmeans(points, n, k, limit, currentTime, qm, outputString);
	printf(outputString);
	fflush(stdout);	
	if (kmeanRet == 1)
		writeDataToFile(outputString);
	free(outputString);
	return kmeanRet;
}

//return 0 if we get no result that q<qn
//else return the outputSizeStr
//if writeFlag==0 the output will be write to the file
int kmeanSlaveResult(int slaveId, int writeFlag) { //run on main process
	MPI_Status status;
	int outputStrSize = 0;
	MPI_Recv(&outputStrSize, 1, MPI_INT, slaveId, 0, MPI_COMM_WORLD, &status); //get the size of the output (string)	
	if (outputStrSize == 0)
		return 0;
	char *outputString = (char *)malloc(sizeof(char)*outputStrSize + 1);
	if (outputString == NULL) {
		printf("Can't malloc outputString kmeanSlaveResult()\n");
		MPI_Abort(MPI_COMM_WORLD, 0);
	}
	MPI_Recv(outputString, outputStrSize, MPI_CHAR, slaveId, 0, MPI_COMM_WORLD, &status); //get str output
	outputString[outputStrSize] = '\0';
	if (writeFlag == 0)
		writeDataToFile(outputString);
	free(outputString);
	return outputStrSize;
}

int master_process(int myid, int numprocs, Point* points, long n, int k, double t, double dt, int limit, double qm) {
	double t1 = MPI_Wtime();
	int finishFlag = 0;
	double currentTime = 0;
	double closeProcess = CLOSE_PROCESS_FLAG; //flag to process to clost it
	int loops = ((int)floor(t / dt)) / NUM_OF_PROCESS;
	int remainder = ((int)floor(t / dt)) % NUM_OF_PROCESS;

	for (int i = 0; i < loops; i++) {
		MPI_Send(&currentTime, 1, MPI_DOUBLE, SLAVE1_PROC, 0, MPI_COMM_WORLD); // send a time to slave
		currentTime += dt;
		MPI_Send(&currentTime, 1, MPI_DOUBLE, SLAVE2_PROC, 0, MPI_COMM_WORLD); // send a time to slave
		currentTime += dt;
		int retMainProcess = kmeanMainResult(points, n, k, currentTime, limit, qm);
		currentTime += dt;

		int strSlave1Size = kmeanSlaveResult(SLAVE1_PROC, 0);
		int strSlave2Size = kmeanSlaveResult(SLAVE2_PROC, strSlave1Size);
		if (retMainProcess != 0 || strSlave1Size != 0 || strSlave2Size != 0) {
			MPI_Send(&closeProcess, 1, MPI_DOUBLE, SLAVE2_PROC, 0, MPI_COMM_WORLD); //close process 2
			MPI_Send(&closeProcess, 1, MPI_DOUBLE, SLAVE1_PROC, 0, MPI_COMM_WORLD); //close process 1
			free(points);
			printf("Master: end! \n");
			printCurrentTime(t1, 0);
			fflush(stdout);
			return 0;
		}
	}
	MPI_Send(&closeProcess, 1, MPI_DOUBLE, SLAVE2_PROC, 0, MPI_COMM_WORLD); //close process 2
	if (remainder == 2) {
		MPI_Send(&currentTime, 1, MPI_DOUBLE, SLAVE1_PROC, 0, MPI_COMM_WORLD); // send a time to slave
		currentTime += dt;
		int retMainProcess = kmeanMainResult(points, n, k, currentTime, limit, qm);
		currentTime += dt;
		int strSlave1Size = kmeanSlaveResult(SLAVE1_PROC, 0);
	}
	else if (remainder == 1) {
		int retMainProcess = kmeanMainResult(points, n, k, currentTime, limit, qm);
		currentTime += dt;
	}
	MPI_Send(&closeProcess, 1, MPI_DOUBLE, SLAVE1_PROC, 0, MPI_COMM_WORLD); //close process 1
	free(points);
	printf("Master: end! \n");
	printCurrentTime(t1, 0);
	fflush(stdout);
	return 0;
}

int slave_process(int myid, int numprocs, Point* points, long n, int k, int limit, double qm) {
	MPI_Status status;
	double currentTime;
	int ret;

	MPI_Recv(&currentTime, 1, MPI_DOUBLE, MAIN_PROCESS, 0, MPI_COMM_WORLD, &status);
	while (currentTime != CLOSE_PROCESS_FLAG) {
		char *output = (char *)malloc(sizeof(char)* LINE_SIZE * (5 + k));
		sprintf(output, "%s", "");
		ret = kmeans(points, n, k, limit, currentTime, qm, output);
		printf(output); // delete it
		fflush(stdout);
		if (ret == 0) {
			MPI_Send(&ret, 1, MPI_INT, MAIN_PROCESS, 0, MPI_COMM_WORLD); //ret 0 bceause we don't find the desirerd qm
		}
		else {
			int outputStrSize = strlen(output);
			MPI_Send(&outputStrSize, 1, MPI_INT, MAIN_PROCESS, 0, MPI_COMM_WORLD); //send strlen(output)
			MPI_Send(output, strlen(output), MPI_CHAR, MAIN_PROCESS, 0, MPI_COMM_WORLD);
		}
		free(output);
		MPI_Recv(&currentTime, 1, MPI_DOUBLE, MAIN_PROCESS, 0, MPI_COMM_WORLD, &status);
	}
	free(points);
	return 0;
}

MPI_Datatype initPointMpiType(Point *points) {
	MPI_Datatype PointType;
	MPI_Datatype type[MPI_DATA_FILEDS_COUNT] = { MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE };
	int blocklen[MPI_DATA_FILEDS_COUNT] = { 1, 1, 1, 1, 1, 1 };

	MPI_Aint disp[MPI_DATA_FILEDS_COUNT];
	MPI_Address(&points[0].x, disp + 0);
	MPI_Address(&points[0].y, disp + 1);
	MPI_Address(&points[0].z, disp + 2);
	MPI_Address(&points[0].vX, disp + 3);
	MPI_Address(&points[0].vY, disp + 4);
	MPI_Address(&points[0].vZ, disp + 5);

	int base = disp[0];
	for (int i = 0; i <MPI_DATA_FILEDS_COUNT; i++)
		disp[i] -= base;

	MPI_Type_create_struct(MPI_DATA_FILEDS_COUNT, blocklen, disp, type, &PointType);
	MPI_Type_commit(&PointType);
	return PointType;
}

Point* initSettings(int myid, long *n, int *k, double *t, int *limit, double *dt, double *qm) {
	Point *points = NULL;
	if (myid == MAIN_PROCESS) {
		points = readWordsFromFile(points, n, k, t, dt, limit, qm);
		*t += 1; //Becuse we needs to check the last value
	}
	MPI_Bcast(n, 1, MPI_LONG, MAIN_PROCESS, MPI_COMM_WORLD);
	MPI_Bcast(k, 1, MPI_INT, MAIN_PROCESS, MPI_COMM_WORLD);
	MPI_Bcast(limit, 1, MPI_INT, MAIN_PROCESS, MPI_COMM_WORLD);
	MPI_Bcast(qm, 1, MPI_DOUBLE, MAIN_PROCESS, MPI_COMM_WORLD);
	if (myid != MAIN_PROCESS)
		points = (Point *)malloc((*n) * sizeof(Point));
	MPI_Datatype PointType = initPointMpiType(points);
	MPI_Bcast(points, *n, PointType, MAIN_PROCESS, MPI_COMM_WORLD);
	return points;
}

int main(int argc, char *argv[]) {
	int  myid, numprocs;
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &myid);
	MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
	if (numprocs != NUM_OF_PROCESS) {
		printf("num of process is:%d instead of:%d\n", numprocs, NUM_OF_PROCESS);
		MPI_Abort(MPI_COMM_WORLD, 0);
	}

	long n;
	int k, limit;
	double t, dt, qm;
	Point *points = NULL;
	points = initSettings(myid, &n, &k, &t, &limit, &dt, &qm);
	if (myid == MAIN_PROCESS) //Master Process
		master_process(myid, numprocs, points, n, k, t, dt, limit, qm);
	else
		slave_process(myid, numprocs, points, n, k, limit, qm);
	MPI_Finalize();
	return finalizeCuda();
}
