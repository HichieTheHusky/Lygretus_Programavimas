#include <iostream>
#include <string>
#include <vector>
#include <thread>
#include <functional>
#include <mutex>
#include <condition_variable>

//for windows
// #include <Windows.h>


using namespace std;

struct monitor {
    mutex mtx;
    condition_variable cv;
    string tekstas = "*";
    bool producerFinished = false;
    int balsesInARow = 0;

    bool add(string name) {

//        cout << name + "  Monitor: item " + tekstas + "\n";  // suletina monitor ir tada geriau matosi veikimas
        if(name == "A") {
            balsesInARow++;
            tekstas += name;
            if(balsesInARow >= 3)
                cv.notify_all();
        }
        else
            {
            unique_lock<mutex> lck(mtx);
            cv.wait(lck, [&] {return balsesInARow >= 3 || producerFinished;});
            if(producerFinished)
                return false;
            tekstas += name;
            balsesInARow = 0;
            }
        return true;
    };

    void setFinished()
    {
     producerFinished = true;
     cv.notify_all();
    }
};

monitor mntr;

void execute(const string &name) {
    int count = 0;
    while (!mntr.producerFinished) {
            if(mntr.add(name))
                count++;

            if(count == 15)
                mntr.setFinished();
    }
//    cout << name + "  : Count : " << count <<  "\n";

}

int main()
{
    vector<string> names = {"C", "B", "A"};
    vector<thread> threads(names.size());
    transform(names.begin(),names.end(), threads.begin(),[](auto& name){return thread(execute, name);});

    while(!mntr.producerFinished)
    {
        cout << "Main thread: " + mntr.tekstas + "\n";
    }
    for_each(threads.begin(),threads.end(), mem_fn(&thread::join));
    cout << "Main thread: Final! \n";
    cout << "Main thread: " + mntr.tekstas + "\n";
    cout << "Main thread: job done! \n";
    return 0;
}