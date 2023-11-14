
#include <pthread.h>
#include <thread>

constexpr int threadCount = 32;

void *worker_thread(void *arg) {
    using namespace std::chrono_literals;

    printf("Hello from thread: #%ld\n", (long) arg);

    std::this_thread::sleep_for(1ms);

    printf("Hello from thread: #%ld\n", (long) arg);

    std::this_thread::sleep_for(1ms);

    printf("Hello from thread: #%ld\n", (long) arg);

    pthread_exit(nullptr);
    return nullptr;
}

int main() {
    pthread_t my_thread[threadCount];

    // Spawn threads
    for (int i = 0; i < threadCount; i++) {
        int ret = pthread_create(&my_thread[i], NULL, &worker_thread, (void *) i);
        if (ret != 0) {
            printf("Error: pthread_create() failed\n");
            exit(EXIT_FAILURE);
        }
    }

    // Join threads
    for (int i = 0; i < threadCount; i++) {
        void *ret_join;
        auto ret = pthread_join(my_thread[i], &ret_join);
        if (ret != 0) {
            perror("pthread_join failed");
            exit(EXIT_FAILURE);
        } else {
            printf("Joined thread : %d\n", i);
        }
    }

    pthread_exit(nullptr);
}