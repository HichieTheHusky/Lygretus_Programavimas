#include <iostream>
#include <iomanip>
#include <string>
#include <fstream>
#include <omp.h>
#include <nlohmann/json.hpp>

using namespace std;
using json = nlohmann::json;

const int THREAD_COUNT = 4;
const int MAIN_MONITOR_SIZE = 10;
const double MIN_FILTER = 50;
const string JSON_DATA = "/home/lukasz/Documents/GitHub/Lygretus_Programavimas/Lab1b/IFF-8-8_ZumarasLukas_L1_dat_1.json"; // 1, 2, 3
const string FINAL_DATA = "/home/lukasz/Documents/GitHub/Lygretus_Programavimas/Lab1b/IFF-8-8_ZumarasLukas_L1_rez.txt";

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
             {"Score", b.Score}};
}

void from_json(const json &j, BenchmarkGPU &p) {
    j.at("Name").get_to(p.Name);
    j.at("MSRP").get_to(p.MSRP);
    j.at("Score").get_to(p.Score);
}

class Monitor {
    private:
        BenchmarkGPU data[MAIN_MONITOR_SIZE];
        int currentIndex;
        bool finished;

    public:
        omp_lock_t ompLock;

        Monitor() {
            currentIndex = 0;
            finished = false;
        }

        bool put(BenchmarkGPU dataLine) {
            if (currentIndex == MAIN_MONITOR_SIZE)
                return false;

            omp_set_lock(&ompLock);
            data[currentIndex++] = dataLine;
            omp_unset_lock(&ompLock);
            return true;
        }

        BenchmarkGPU get(bool &success) {
            omp_set_lock(&ompLock);
            if (currentIndex == 0) {
                omp_unset_lock(&ompLock);
                BenchmarkGPU dummy;
                success = false;
                return dummy;
            }
            BenchmarkGPU new_data = data[currentIndex-- -1];
            omp_unset_lock(&ompLock);
            success = true;
            return new_data;
        }
        void setfinished ()
        {
            finished = true;
        }
        bool getStatus()
        {
            if(currentIndex == 0)
            {
                return finished;
            }
            else
                return false;
        }
};

struct results {
    private:
        BenchmarkGPU filteredResults[25];
        int size = 0;
    public:
        omp_lock_t ompLock;

        void addSortedBenchmark(BenchmarkGPU newest) {
            omp_set_lock(&ompLock);
            for (int i = 0; i < size; i++) {
                if (filteredResults[i].Performance < newest.Performance) {
                    for (int j = size++; j > i; j--) {
                        filteredResults[j] = filteredResults[j - 1];
                    }
                    filteredResults[i] = newest;
                    omp_unset_lock(&ompLock);
                    return;
                }
            }
            filteredResults[size++] = newest;
            omp_unset_lock(&ompLock);
        };

        string takePrint(int i)
        {
            return filteredResults[i].toString() + "\n";
        }

        int ElementCount()
        {
            return size;
        }
};

Monitor DataMonitor;
results SortedResultMonitor;

void execute(const string &name) {
    while (!DataMonitor.getStatus())
    {
        bool addedSuccessfully = false;
//        int failure = 0;
        BenchmarkGPU data;
        while (!addedSuccessfully && !DataMonitor.getStatus()) {
            data = DataMonitor.get(addedSuccessfully);
//            if (failure != 0)
//                cout << "thread " + name + ": failed for " + to_string(failure) + "times\n";
//            failure++;
        }
        cout << "thread " + name + ": took " + data.Name + "\n";
        if(data.MSRP != -1)
        {
            data.Performance = calculateNew(data.MSRP,data.Score);
            if(data.Performance < MIN_FILTER)
                SortedResultMonitor.addSortedBenchmark(data);
        }
    }
}

void printResults() {
    ofstream file;
    file.open(FINAL_DATA);
    file << setw(45) << "Name" << " | " << setw(6) << "MSRP" << " | " << setw(8) << "Score" << " | " << setw(8) << "Value for $" << endl;
    file << string(85, '-') << endl;
    for (int i = 0; i < SortedResultMonitor.ElementCount(); i++)
        file << SortedResultMonitor.takePrint(i);
    file.close();
}

int main() {
    cout << "\033[1;32m Lukas Zumaras IFF-8/8 \033[0m\n" ;

    BenchmarkGPU data[25];

    omp_init_lock(&DataMonitor.ompLock);
    omp_init_lock(&SortedResultMonitor.ompLock);

    // 1) Nuskaitomas duomenu failas i lokalu masyva
    cout << "\033[1;31m Main Thread is getting json data to local array \033[0m\n" ;
    ifstream i(JSON_DATA);
    json j2;
    i >> j2;
    for (int k = 0; k < j2.size(); ++k) {
        data[k] = j2[k];
    }
    auto data_count = j2.size();

    // 2) Paleidziamas pasirinktas kiekis giju, jos vykdo pasirinkta operacija
    #pragma omp parallel num_threads(THREAD_COUNT + 1)
    {
        if (omp_get_thread_num() == 0) {
            // 3) I duomenu struktura, is kurios gijos ims duomenis, irasomi nuskaityti duomenys
            cout << "\033[1;31m Main Thread is putting data into monitor \033[0m\n" ;
            int index = 0;
            while (index < data_count) {
                while (!DataMonitor.put(data[index])) {
                    cout << "\033[1;31m failed to add BenchmarkGPU \033[0m\n" ;
                }
                cout << "main thread: added " << index + 1<< ". " + data[index].Name + "\n";
                index++;
            }
            cout << "\033[1;31m Main Thread putting is finished \033[0m\n" ;
            DataMonitor.setfinished();
        }
        else {
            execute(to_string(omp_get_thread_num()));
        }
    }
    // 4) Laukiama kol visos gijos baigs darba tada sunaikinami uzraktai
    omp_destroy_lock(&DataMonitor.ompLock);
    omp_destroy_lock(&SortedResultMonitor.ompLock);

    // 5) Duomenys isvedami i tekstini faila lentele
    cout << "\033[1;31m Printing table to txt \033[0m\n" ;
    printResults();

    cout << "\033[1;32m Program finish \033[0m\n" ;
    return 0;
}