#include "cuda_runtime.h"
#include <cuda.h>
#include <cstdio>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <vector>
#include "device_launch_parameters.h"

using namespace std;

const int MAX_STRING_LENGTH = 256;
const int THREADS = 3;

const string DATA_FILE = "/home/lukasz/Documents/GitHub/Lygretus_Programavimas/lab3_cuda/IFF-8-8_ZumarasLukas_L1_dat_1.txt"; // 1, 2, 3
const string REZ_FILE = "/home/lukasz/Documents/GitHub/Lygretus_Programavimas/lab3_cuda/IFF-8-8_ZumarasLukas_L1_rez.txt"; // 1, 2, 3

struct BenchmarkGPU {
    char Name[MAX_STRING_LENGTH];
    int MSRP = -1;
    double Score = -1;
    char result[MAX_STRING_LENGTH+2];

    string toString() {
        stringstream ss;
        ss << setw(45) << Name << " | " << setw(6) << MSRP << " | " << setw(8) << Score << " | " << setw(12) << result;
        return ss.str();
    }
};

double calculateNew(int x, double y) {
    return (x / y);
}

void readGPUFile(BenchmarkGPU *data);
void write_results_to_file(BenchmarkGPU* data, int n, const string file_path, const string title);

__global__ void sum_on_gpu(BenchmarkGPU* gpus, int* count, int* n, int* chunk_size, BenchmarkGPU* results);
__device__ void gpu_memset(char* dest, int add);
__device__ void gpu_strcat(char* dest, char* src, int offset);

int main() {
    // Host
    int n = 25;
    BenchmarkGPU data[n];
    readGPUFile(data);
    BenchmarkGPU results[n];
    int chunk_size = n / THREADS;
    int count = 0;
    char* sresults[25];

    // GPU
    BenchmarkGPU* d_all_gpus;
    int* d_count;
    int* d_n;
    int* d_chunk_size;
    BenchmarkGPU* d_results;
    char** d_sresults;


    // Memory allocation for GPU
    cudaMalloc((void**)&d_all_gpus, n * sizeof(BenchmarkGPU));
    cudaMalloc((void**)&d_results, n * sizeof(BenchmarkGPU));
    cudaMalloc((void**)&d_count, sizeof(int));
    cudaMalloc((void**)&d_n, sizeof(int));
    cudaMalloc((void**)&d_chunk_size, sizeof(int));

    // Copies memory from CPU to GPU
    cudaMemcpy(d_all_gpus, data, n * sizeof(BenchmarkGPU), cudaMemcpyHostToDevice);
    cudaMemcpy(d_count, &count, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_n, &n, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_chunk_size, &chunk_size, sizeof(int), cudaMemcpyHostToDevice);

    sum_on_gpu<<<1,THREADS>>>(d_all_gpus, d_count, d_n, d_chunk_size, d_results);
    cudaDeviceSynchronize();

    cudaMemcpy(&results, d_results, n * sizeof(BenchmarkGPU), cudaMemcpyDeviceToHost);
    cudaMemcpy(&count, d_count, 1, cudaMemcpyDeviceToHost);
    cudaFree(d_all_gpus);
    cudaFree(d_count);
    cudaFree(d_n);
    cudaFree(d_chunk_size);
    cudaFree(d_results);

    cout << "Found results: " << count << endl;
    cout << "Finished" << endl;
    write_results_to_file(results, count, REZ_FILE, "A dalies rezultatai");
    return 0;
}
/**
 * GPU
 * Sums gpus list chunk data properties
 * @param gpus BenchmarkGPUs list
 * @param count BenchmarkGPUs list size
 * @param chunk_size Summed items per thread
 * @param results Summed chunk results
 */
__global__ void sum_on_gpu(BenchmarkGPU* gpus, int* count, int* n, int* chunk_size, BenchmarkGPU* results) {
    int start_index = threadIdx.x * *chunk_size;
    int end_index = start_index + 1 * *chunk_size;
    if (threadIdx.x == blockDim.x -1)
        end_index = *n;


        printf("Thread: %d Start Index: %d End Index: %d\n", threadIdx.x, start_index, end_index);
    for (int i = start_index; i < end_index; ++i) {
        BenchmarkGPU tmp;
        gpu_memset(tmp.Name,0);
        gpu_memset(tmp.result,2);
        tmp.MSRP = 0;
        tmp.Score = 0.0;

        gpu_strcat(tmp.Name, gpus[i].Name, 0);
        tmp.Score = gpus[i].Score;
        tmp.MSRP = gpus[i].MSRP;


        double my_number = tmp.MSRP / tmp.Score;

        char tmp_res[256+2];
        gpu_memset(tmp_res, 2);
        tmp_res[0] = 'F';
        tmp_res[1] = '-';
        if(my_number < 70)
            tmp_res[0] = 'E';
        if(my_number < 60)
            tmp_res[0] = 'D';
        if(my_number < 50)
            tmp_res[0] = 'C';
        if(my_number < 40)
            tmp_res[0] = 'B';
        if(my_number < 30)
            tmp_res[0] = 'A';

        gpu_strcat(tmp_res, gpus[i].Name, 2);
        printf("Thread: %d Brand: %d\n", threadIdx.x, tmp_res);

        gpu_strcat(tmp.result, tmp_res,0);

        if(tmp.result[0] < 'F')
        {
            int index = atomicAdd(count, 1);
            results[index] = tmp;
        }

//            printf("Thread: %d Index: %d Brand: %s Make Year: %d Mileage: %f\n", threadIdx.x, index, results[index].Name, results[index].Score, results[index].MSRP);
    }
}

/**
 * Appends char array to other char array
 * @param dest Destination array
 * @param src Source array
 */
__device__ void gpu_strcat(char* dest, char* src, int offset) {
    int i = 0;
    do {
        dest[offset + i] = src[i];}
    while (src[i++] != 0 && i + offset != MAX_STRING_LENGTH+offset);
}

/**
 * Zeroes all char memory
 * @param dest Char array
 */
__device__ void gpu_memset(char* dest, int add) {
    for (int i = 0; i < MAX_STRING_LENGTH + add; ++i) {
        dest[i] = 0;
    }
}

void readGPUFile(BenchmarkGPU *data)
{
    string line;
    ifstream myfile;
    myfile.open(DATA_FILE);
    if(!myfile.is_open()) {
        perror("Error open");
        exit(EXIT_FAILURE);
    }

    int ch = 0;
    int count = 0;
    while(getline(myfile, line)) {
        string::size_type pos;
        pos=line.find(' ',0);
        line = line.substr(pos+1);
        switch (ch) {
            case 0:
                strcpy(data[count].Name, line.c_str());
                break;
            case 1:
                data[count].MSRP =  stoi(line);
                break;
            case 2:
                data[count].Score =  stoi(line);
                count++;
                ch = -1;
                break;
        }
        ch++;
    }
}
/**
 * Writes given monitor cars formatted in table to file
 * @param cars Cars list
 * @param file_path Result file path
 * @param title Results table title
 */
void write_results_to_file(BenchmarkGPU* gpus, int n, const string file_path, const string title) {
    ofstream file;
    file.open(file_path);
    file << setw(80) << title << endl
         << "------------------------------------------------------------------------------------------------------------------------"
         << endl
         << setw(45) << "Name" << " | " << setw(6) << "MSRP" << " | " << setw(8) << "Score" << " | " << setw(20) << "Result" << endl
         << "------------------------------------------------------------------------------------------------------------------------"
         << endl;
    for (int i = 0; i < n; ++i) {
        file << gpus[i].toString() << endl;
    }

    file << endl << endl << endl;
}
