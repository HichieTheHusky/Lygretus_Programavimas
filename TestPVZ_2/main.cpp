#include <iostream>
#include <iomanip>
#include <string>
#include <fstream>
#include <nlohmann/json.hpp>
#include <mpi/mpi.h>

using namespace std;
using namespace MPI;

const int DATA_LIMIT = 20;
const int Stop = -1;

int WORLD_RANK;
int WORLD_SIZE;

void execute_worker() {
    int number;

    if(WORLD_RANK == 3)
        number = 0;

    if(WORLD_RANK == 4)
        number = 11;

    while (true)
    {
        int next_action;
        MPI_Send(&number,1,MPI_INT,0,0,MPI_COMM_WORLD);
        MPI_Recv(&next_action,1,MPI_INT,0,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);

        if(next_action == Stop)
            break;

        number++;
    }
}

void execute_data(){
    int number;
    int count = 0;

    while(true)
    {
        MPI_Status status;

        MPI_Recv(&number,1,MPI_INT,MPI_ANY_SOURCE,0,MPI_COMM_WORLD,&status);
        MPI_Send(&number,1,MPI_INT,status.MPI_SOURCE,0,MPI_COMM_WORLD);


        if(number % 2 == 0)
        {
            // lyginis
            MPI_Send(&number,1,MPI_INT,2,0,MPI_COMM_WORLD);

        }
        else
        {
            // nelyginis
            MPI_Send(&number,1,MPI_INT,1,0,MPI_COMM_WORLD);
        }
        count++;
        if(count == DATA_LIMIT)
            break;
    }

    MPI_Send(&Stop,1,MPI_INT,3,0,MPI_COMM_WORLD);
    MPI_Send(&Stop,1,MPI_INT,4,0,MPI_COMM_WORLD);
    MPI_Send(&Stop,1,MPI_INT,1,0,MPI_COMM_WORLD);
    MPI_Send(&Stop,1,MPI_INT,2,0,MPI_COMM_WORLD);
}


void execute_results(){
    int numbers[DATA_LIMIT];
    int count = 0;
    while(true)
    {
        int number;
        MPI_Recv(&number,1,MPI_INT,0,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);

        if(number == Stop)
            break;

        numbers[count] = number;
        count++;
    }
    for (int i = 0; i < count; ++i) {
        cout << "Thread " << WORLD_RANK << " : " << numbers[i] << endl;
    }
}

int main() {
    MPI_Init(NULL,NULL);
    MPI_Comm_rank(MPI_COMM_WORLD, &WORLD_RANK);
    MPI_Comm_size(MPI_COMM_WORLD, &WORLD_SIZE);


    if(WORLD_RANK == 0)
        execute_data();

    if(WORLD_RANK == 1)
        execute_results();

    if(WORLD_RANK == 2)
        execute_results();

    if(WORLD_RANK == 3)
        execute_worker();

    if(WORLD_RANK == 4)
        execute_worker();

    cout << WORLD_RANK << "\033[1;32m thread finish \033[0m\n" ;
    MPI_Finalize();
    return 0;


}