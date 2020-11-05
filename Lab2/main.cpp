#include <iostream>
#include <iomanip>
#include <string>
#include <fstream>
#include <nlohmann/json.hpp>
#include <mpi/mpi.h>

using namespace std;
using namespace MPI;
using json = nlohmann::json;

const int DATA_SIZE = 25;
const double MIN_FILTER = 50;
const string JSON_DATA = "/home/lukasz/Documents/GitHub/Lygretus_Programavimas/Lab2/IFF-8-8_ZumarasLukas_L1_dat_1.json"; // 1, 2, 3
const string FINAL_DATA = "/home/lukasz/Documents/GitHub/Lygretus_Programavimas/Lab2/IFF-8-8_ZumarasLukas_L1_rez.txt";

int WORLD_RANK;
int WORLD_SIZE;
int WORKER_THREADS = 2;

struct BenchmarkGPU {
    string Name;
    int MSRP = -1;
    double Score = -1;
    double Performance = -1;

    string toString() {
        stringstream ss;
        ss << setw(45) << Name << " | " << setw(6) << MSRP << " | " << setw(8) << Score << " | " << setw(8) << Performance;
        return ss.str();
    }
};

double calculateNew(int x, double y) {
    return (x / y);
}

void to_json(json &j, const BenchmarkGPU &b) {
    j = json{{"Name",  b.Name},
             {"MSRP",  b.MSRP},
             {"Score", b.Score},
             {"Performance", b.Performance}};
}

void from_json(const json &j, BenchmarkGPU &p) {
    j.at("Name").get_to(p.Name);
    j.at("MSRP").get_to(p.MSRP);
    j.at("Score").get_to(p.Score);
}

BenchmarkGPU from_json_mod(string json_string){
    auto parsed = json::parse(json_string);
    BenchmarkGPU tmp;
    tmp.Name = parsed["Name"];
    tmp.MSRP = parsed["MSRP"];
    tmp.Score = parsed["Score"];
    tmp.Performance = parsed["Performance"];
    return tmp;
}


void execute_worker() {
    while (true)
    {
        int k = WORLD_RANK;
        while( k == WORLD_RANK)
        {
            MPI_Send(&k,1,MPI_INT,1,0,MPI_COMM_WORLD);
            MPI_Recv(&k,1,MPI_INT,MPI_ANY_SOURCE,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
        }
        if(k == -2)
            break;

        int size;
        MPI_Recv(&size,1,MPI_INT,MPI_ANY_SOURCE,1,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
        char serialized[size];
        MPI_Recv(serialized,size,MPI_CHAR,MPI_ANY_SOURCE,2,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
        BenchmarkGPU data = json::parse(string(serialized, 0, static_cast<unsigned long>(size)));
        data.Performance = calculateNew(data.MSRP, data.Score);

        if(data.Performance < MIN_FILTER)
        {
            json j2 = data;
            string message = j2.dump();
            int serialized_size = static_cast<int>(message.size());
            const char* serialized_char = message.c_str();
            MPI_Send(&serialized_size,1,MPI_INT,2,1,MPI_COMM_WORLD);
            MPI_Send(serialized_char,serialized_size,MPI_CHAR,2,2,MPI_COMM_WORLD);
        }
    }

    int ending = -1;
    MPI_Send(&ending,1,MPI_INT,2,1,MPI_COMM_WORLD);
}

void execute_data(){
    int max_size = 10;
    BenchmarkGPU data[10];
    int current_size = 0;
    int lifetime_size = 0;
    int waiting_worker = true;
    int tas = 0;

    while (lifetime_size != DATA_SIZE){
        int source;
        MPI_Recv(&source,1,MPI_INT,MPI_ANY_SOURCE,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);

        if(source == 0 && current_size < 10)
        {
            int k = -1;
            MPI_Send(&k,1,MPI_INT,source,0,MPI_COMM_WORLD);
            int size;
            MPI_Recv(&size,1,MPI_INT,source,1,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
            char serialized[size];
            MPI_Recv(serialized,size,MPI_CHAR,source,2,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
            data[current_size] = json::parse(string(serialized, 0, static_cast<unsigned long>(size)));
            current_size++;
        }
        else if (source == 0)
        {
            int k = source;
            MPI_Send(&k,1,MPI_INT,0,0,MPI_COMM_WORLD);
        }
        else if(source != 0 && current_size != 0)
        {
            lifetime_size++;
            int k = -1;

            MPI_Send(&k,1,MPI_INT,source,0,MPI_COMM_WORLD);
            current_size--;
            json j2 = data[current_size];
            data[current_size] = *new BenchmarkGPU;
            string message = j2.dump();
            int serialized_size = static_cast<int>(message.size());
            const char* serialized_char = message.c_str();
            MPI_Send(&serialized_size,1,MPI_INT,source,1,MPI_COMM_WORLD);
            MPI_Send(serialized_char,serialized_size,MPI_CHAR,source,2,MPI_COMM_WORLD);
        }
        else if (source != 0)
        {
            int k = source;
            MPI_Send(&k,1,MPI_INT,source,0,MPI_COMM_WORLD);
        }
    }

    int source;
    int t = -2;
    MPI_Recv(&source,1,MPI_INT,MPI_ANY_SOURCE,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
    MPI_Send(&t,1,MPI_INT,source,0,MPI_COMM_WORLD);
    MPI_Recv(&source,1,MPI_INT,MPI_ANY_SOURCE,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
    MPI_Send(&t,1,MPI_INT,source,0,MPI_COMM_WORLD);
}

void execute_results(){
    BenchmarkGPU data[26];
    int current_size = 0;
    int ending_size = 0;
    while(true)
    {
        MPI_Status status;
        int size;
        MPI_Recv(&size,1,MPI_INT,MPI_ANY_SOURCE,1,MPI_COMM_WORLD,&status);
        if(size == -1)
        {
            ending_size++;
            if(ending_size == WORKER_THREADS)
                break;
        } else{
            int source = status.MPI_SOURCE;
            char serialized[size];
            MPI_Recv(serialized,size,MPI_CHAR,source,2,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
            data[current_size] = from_json_mod(string(serialized, 0, static_cast<unsigned long>(size)));
            current_size++;
        }
    }
    for (int i = 0; i < current_size; ++i) {
        json j2 = data[i];
        string message = j2.dump();
        int serialized_size = static_cast<int>(message.size());
        const char* serialized_char = message.c_str();
        MPI_Send(&serialized_size,1,MPI_INT,0,1,MPI_COMM_WORLD);
        MPI_Send(serialized_char,serialized_size,MPI_CHAR,0,2,MPI_COMM_WORLD);
    }
    int serialized_size = -1;
    MPI_Send(&serialized_size,1,MPI_INT,0,1,MPI_COMM_WORLD);
}

int main() {
    MPI_Init(NULL,NULL);
    MPI_Comm_rank(MPI_COMM_WORLD, &WORLD_RANK);
    MPI_Comm_size(MPI_COMM_WORLD, &WORLD_SIZE);


    if(WORLD_RANK == 0)
    {
        cout << "\033[1;32m Lukas Zumaras IFF-8/8 \033[0m\n" ;

        printf("Main thread - rank %d out of %d processors\n",WORLD_RANK, WORLD_SIZE);
        BenchmarkGPU data[25];

        // 1) Nuskaitomas duomenu failas i lokalu masyva
        cout << "\033[1;31m Main Thread is getting json data to local array \033[0m\n" ;
        ifstream i(JSON_DATA);
        json j2;
        i >> j2;
        for (int k = 0; k < j2.size(); ++k) {
            data[k] = j2[k];
        }

        // 2) Siunciama duomenys i duomenu thread
        for (int j = 0; j < DATA_SIZE; ++j) {

            // 2a) prasome leidimo siusti duomenys
            int k = WORLD_RANK;
            while( k == WORLD_RANK)
            {
                MPI_Send(&k,1,MPI_INT,1,0,MPI_COMM_WORLD);
                MPI_Recv(&k,1,MPI_INT,MPI_ANY_SOURCE,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
            }

            // 2b) Siunciama duomenys i duomenu thread
            string message = j2[j].dump();
            int serialized_size = static_cast<int>(message.size());
            const char* serialized_char = message.c_str();
            MPI_Send(&serialized_size,1,MPI_INT,1,1,MPI_COMM_WORLD);
            MPI_Send(serialized_char,serialized_size,MPI_CHAR,1,2,MPI_COMM_WORLD);
        }

        // 3) spausdinime i faila
        ofstream file;
        file.open(FINAL_DATA);
        file << setw(45) << "Name" << " | " << setw(6) << "MSRP" << " | " << setw(8) << "Score" << " | " << setw(8) << "Value for $" << endl;
        file << string(85, '-') << endl;
        while (true)
        {
            int size;
            MPI_Recv(&size,1,MPI_INT,MPI_ANY_SOURCE,1,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
            if(size == -1)
                break;
            char serialized[size];
            MPI_Recv(serialized,size,MPI_CHAR,MPI_ANY_SOURCE,2,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
            BenchmarkGPU data = from_json_mod(string(serialized, 0, static_cast<unsigned long>(size)));
            file << data.toString() << endl;
        }
        file.close();
    }

    if(WORLD_RANK == 1)
       execute_data();

    if(WORLD_RANK == 2)
        execute_results();

    if(WORLD_RANK < 6 && WORLD_RANK > 2)
        execute_worker();

    cout << WORLD_RANK << "\033[1;32m thread finish \033[0m\n" ;
    MPI_Finalize();
    return 0;


}