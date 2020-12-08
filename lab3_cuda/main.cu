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
};

void readGPUFile(BenchmarkGPU *data);

__global__ void sum_on_gpu(BenchmarkGPU* gpus, int* count, int* n, int* chunk_size, char* results);
__device__ void gpu_memset(char* dest, int add);
__device__ int gpu_strcat(char* dest, char* src, int offset, bool nLine);

int main() {
    // Host
    int n = 25;
    BenchmarkGPU data[n];
    readGPUFile(data);
    char sresults[n * MAX_STRING_LENGTH];
    int chunk_size = n / THREADS;
    int count = 0;

    // GPU
    BenchmarkGPU* d_all_gpus;
    int* d_count;
    int* d_n;
    int* d_chunk_size;
    char* d_sresults;

    // Memory allocation for GPU
    cudaMalloc((void**)&d_all_gpus, n * sizeof(BenchmarkGPU));
    cudaMalloc((void**)&d_sresults, n * sizeof(char) * MAX_STRING_LENGTH);
    cudaMalloc((void**)&d_count, sizeof(int));
    cudaMalloc((void**)&d_n, sizeof(int));
    cudaMalloc((void**)&d_chunk_size, sizeof(int));

    // Copies memory from CPU to GPU
    cudaMemcpy(d_all_gpus, data, n * sizeof(BenchmarkGPU), cudaMemcpyHostToDevice);
    cudaMemcpy(d_count, &count, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_n, &n, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_chunk_size, &chunk_size, sizeof(int), cudaMemcpyHostToDevice);

    sum_on_gpu<<<1,THREADS>>>(d_all_gpus, d_count, d_n, d_chunk_size, d_sresults);
    cudaDeviceSynchronize();

    // Copies memory from GPU to CPU
    cudaMemcpy(&sresults, d_sresults, n * MAX_STRING_LENGTH * sizeof(char), cudaMemcpyDeviceToHost);
    cudaMemcpy(&count, d_count, 1, cudaMemcpyDeviceToHost);
    // Free memory
    cudaFree(d_all_gpus);
    cudaFree(d_count);
    cudaFree(d_n);
    cudaFree(d_chunk_size);
    cudaFree(d_sresults);

    cout << "Printing" << endl;
    ofstream file;
    file.open(REZ_FILE);
    file << "Resultatai" << endl;
    file << "" << endl;
    if(count == 0)
        file << "Neivienas objektas nepraejo filtro" << endl;
    else
        file << sresults << endl;

    cout << "Finished" << endl;
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
__global__ void sum_on_gpu(BenchmarkGPU* gpus, int* count, int* n, int* chunk_size, char* results) {
    int start_index = threadIdx.x * *chunk_size;
    int end_index = start_index + 1 * *chunk_size;
    if (threadIdx.x == blockDim.x -1)
        end_index = *n;
    printf("Thread: %d Start Index: %d End Index: %d\n", threadIdx.x, start_index, end_index);
    for (int i = start_index; i < end_index; ++i) {
        double my_number = gpus[i].MSRP / gpus[i].Score;
        char tmp_res[256];
        gpu_memset(tmp_res, 0);

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

        int cou = 2;
        cou += gpu_strcat(tmp_res, gpus[i].Name, 2, true);

        if(tmp_res[0] < 'E')
        {
            int index = atomicAdd(count, cou);
            gpu_strcat(results, tmp_res,index, false);
//            printf("Thread: %d Index: %d Result: %s ", threadIdx.x, cou, tmp.result);
        }

    }
}

__device__ int gpu_strcat(char* dest, char* src, int offset, bool nLine) {
    int i = 0;
    do {
        if(src[i] == 0 )
        {
            if(nLine)
            {
                dest[offset + i] = '\n';
                return i+1;
            }
            return i;
        }
        else
        dest[offset + i] = src[i];
        i++;}
    while (i != MAX_STRING_LENGTH);
}

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

