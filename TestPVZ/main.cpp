#include <iostream>
#include <string>
#include <vector>
#include <thread>
#include <functional>
#include <mutex>
#include <condition_variable>

//for windows
// #include <Windows.h>
#include <unistd.h>

using namespace std;

struct monitor {
    mutex mtx;
    condition_variable cv;
    string tekstas = "*";
    bool producerFinished = false;
    int balsesInARow = 0;

    bool add(string name) {
//        usleep(1000000);
        unique_lock<mutex> lck(mtx);

        if(name !="A")
        {
            cout << name + "  Monitor: pries " + "\n";
            cv.wait(lck, [&] {return balsesInARow >= 3 || producerFinished;});
            cout << name + "  Monitor: po " + "\n";
            if(producerFinished)
                return false;
            tekstas += name;
            balsesInARow = 0;
        }
        else {
            balsesInARow++;
            tekstas += name;
            if(balsesInARow >= 3)
            {
                cout << name + "  Monitor: notify " + "\n";
                cv.notify_all();
            }
        }
        cout << name + "  Monitor: item " + tekstas + "\n";  // suletina monitor ir tada geriau matosi veikimas


        return true;
    };

    void setFinished()
    {
     producerFinished = true;
//     cv.notify_all();
    }
};

monitor mntr;

void execute(const string &name) {
    int count = 0;
    while (!mntr.producerFinished) {
            if(mntr.add(name))
                count++;

            if(count == 100)
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
//        cout << "Main thread: " + mntr.tekstas + "\n";
    }
    for_each(threads.begin(),threads.end(), mem_fn(&thread::join));
    cout << "Main thread: Final! \n";
    cout << "Main thread: " + mntr.tekstas + "\n";
    cout << "Main thread: job done! \n";
    return 0;
}