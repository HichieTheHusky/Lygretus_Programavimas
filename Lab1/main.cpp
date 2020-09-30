#include <iostream>
#include <string>
#include <vector>
#include <thread>
#include <algorithm>
#include <functional>
#include <nlohmann/json.hpp>
#include <fstream>
#include <iomanip>
#include <mutex>
#include <condition_variable>

using namespace std;
using json = nlohmann::json;

const int MAIN_MONITOR_SIZE = 10;
const double MIN_FILTER = 50;
const string JSON_DATA = "/home/lukasz/Documents/GitHub/Lygretus_Programavimas/Lab1/IFF-8-8_ZumarasLukas_L1_dat_1.json"; // 1, 2, 3
const string FINAL_DATA = "/home/lukasz/Documents/GitHub/Lygretus_Programavimas/Lab1/IFF-8-8_ZumarasLukas_L1_rez.txt";

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
    BenchmarkGPU data[MAIN_MONITOR_SIZE];
    mutex lock;
    condition_variable cv1;
    condition_variable cv2;
    int currentIndex;
    bool finished;

public:
    Monitor() {
        currentIndex = 0;
        finished = false;
    }

    void put(BenchmarkGPU dataLine) {
        unique_lock<mutex> guard(lock);
        cv2.wait(guard, [&] {return currentIndex != MAIN_MONITOR_SIZE;});
        data[currentIndex++] = dataLine;
        cv1.notify_one();
    }

    BenchmarkGPU get() {
        unique_lock<mutex> guard(lock);
        cv1.wait(guard, [&] {return currentIndex != 0 || finished;});
        if (currentIndex == 0) {
            BenchmarkGPU dummy;
            return dummy;
        }
        currentIndex--;
        BenchmarkGPU new_data = data[currentIndex];
        cv2.notify_one();
        return new_data;
    }
    void setfinished ()
    {
        finished = true;
        cv1.notify_all();
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
    mutex lock;
    BenchmarkGPU filteredResults[25];
    int size = 0;

    void addSortedBenchmark(BenchmarkGPU newest) {
        lock.lock();
        for (int i = 0; i < size; i++) {
            if (filteredResults[i].Performance < newest.Performance) {
                for (int j = size++; j > i; j--) {
                    filteredResults[j] = filteredResults[j - 1];
                }
                filteredResults[i] = newest;
                lock.unlock();
                return;
            }
        }
        filteredResults[size++] = newest;
        lock.unlock();
    };
};

// testing monitor
Monitor DataMonitor;
results SortedResultMonitor;

void execute(const string &name) {
    while (!DataMonitor.getStatus())
    {
        auto test = DataMonitor.get();
        cout << "thread " + name + ": took " + test.Name + "\n";
        if(test.MSRP != -1)
        {
            test.Performance = calculateNew(test.MSRP,test.Score);
            if(test.Performance < MIN_FILTER)
            SortedResultMonitor.addSortedBenchmark(test);
        }
    }
}

void printResults() {
    ofstream file;
    file.open(FINAL_DATA);

    file << setw(45) << "Name" << " | " << setw(6) << "MSRP" << " | " << setw(8) << "Score" << " | " << setw(8) << "Value for $" << endl;
    file << string(85, '-') << endl;
    for (int i = 0; i < SortedResultMonitor.size; i++)
        file << SortedResultMonitor.filteredResults[i].toString() + "\n";

    file.close();
}

int main() {
    cout << "\033[1;32m Lukas Zumaras IFF-8/8 \033[0m\n" ;

    BenchmarkGPU data[25];

    // 1) Nuskaitomas duomenu failas i lokalu masyva
    cout << "\033[1;31m Main Thread is getting json data to local array \033[0m\n" ;
    ifstream i(JSON_DATA);
    json j2;
    i >> j2;
    for (int k = 0; k < j2.size(); ++k) {
        data[k] = j2[k];
    }

    // 2) Paleidziamas pasirinktas kiekis giju, jos vykdo pasirinkta operacija
    cout << "\033[1;31m Main Thread is starting more threads \033[0m\n" ;
    vector<string> names = {"1st thread", "2nd thread", "3rd thread", "4th thread"};
    vector<thread> threads(names.size());
    transform(names.begin(),names.end(), threads.begin(),[](auto& name){return thread(execute, name);});

    // 3) I duomenu struktura, is kurios gijos ims duomenis, irasomi nuskaityti duomenys
    cout << "\033[1;31m Main Thread is putting data into monitor \033[0m\n" ;
    for (int k = 0; k < j2.size(); ++k) {
        DataMonitor.put(data[k]);
        cout << "main thread: added " << k + 1<< ". " + data[k].Name+ "\n";
    }
    DataMonitor.setfinished();
    cout << "\033[1;31m Main Thread putting is finished \033[0m\n" ;

    // 4) Laukiama kol visos gijos baigs darba
    cout << "\033[1;31m Waiting for Threads to finish \033[0m\n" ;
    for_each(threads.begin(),threads.end(), mem_fn(&thread::join));
    cout << "\033[1;31m Threads finished \033[0m\n" ;

    // 5) Duomenys isvedami i tekstini faila lentele
    cout << "\033[1;31m Printing table to txt \033[0m\n" ;
    printResults();

    cout << "\033[1;32m Program finish \033[0m\n" ;
    return 0;
}
