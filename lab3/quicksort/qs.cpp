
#include <iostream>
#include <vector>
#include <algorithm>
#include <cstdlib>
#include <ctime>

#include <omp.h>
#include <execution>

#include "qsJob.h"
#include "scope_profile.h"

int partition(std::vector<int> &data, int low, int high) {

    int pivot = data[low];
    int i = low - 1;
    int j = high + 1;
    while (true) {
        do {
            i++;
        } while (data[i] < pivot);

        do {
            j--;
        } while (data[j] > pivot);

        if (i >= j)
            return j;

        std::swap(data[i], data[j]);
    }

}

void quickSortParallel(std::vector<int> &arr, int low, int high);

void quickSortSequential(std::vector<int> &arr, int low, int high) {
    if (low < high) {
        int pi = partition(arr, low, high);

        quickSortSequential(arr, low, pi);
        quickSortSequential(arr, pi + 1, high);

    }
}

void quickSortParallel(std::vector<int> &arr, int low, int high) {
    if (low < high) {
        // Partition the array
        int pi = partition(arr, low, high);

        constexpr int threshold = 4096;
        if (high - low < threshold) {
            quickSortSequential(arr, low, pi);
            quickSortSequential(arr, pi + 1, high);
        } else {
#pragma omp parallel num_threads(2)
            {
#pragma omp sections
                {
#pragma omp section
                    quickSortParallel(arr, low, pi);
#pragma omp section
                    quickSortParallel(arr, pi + 1, high);
                }
            }
        }
    }
}

void quicksortFullyParallelIntoSeq(int *arr, int left, int right) {
    int i = left;
    int j = right;
    int pivot = arr[left];

    {
        while (i <= j) {
            while (arr[i] < pivot) {
                i++;
            }
            while (arr[j] > pivot) {
                j--;
            }
            if (i <= j) {
                int t = arr[i];
                arr[i] = arr[j];
                arr[j] = t;
                i++;
                j--;
            }
        };
    }

    if (left < j)
        quicksortFullyParallelIntoSeq(arr, left, j);

    if (i < right)
        quicksortFullyParallelIntoSeq(arr, i, right);

}

void quickSortParallelDivideThreads(int *arr, int lenArray, int numThreads) {
    int pivot, i, j, temp;

    if (numThreads <= 1) {
        quicksortFullyParallelIntoSeq(arr, 0, lenArray);
    } else {
        i = 0;
        j = lenArray;
        pivot = arr[lenArray / 2];

        {
            while (i <= j) {
                while (arr[i] < pivot) {
                    i++;
                }
                while (arr[j] > pivot) {
                    j--;
                }
                if (i <= j) {
                    int t = arr[i];
                    arr[i] = arr[j];
                    arr[j] = t;
                    i++;
                    j--;
                }
            };
        }

#pragma omp parallel sections
        {
#pragma omp section
            {
                quickSortParallelDivideThreads(arr, i, numThreads / 2);
            }

#pragma omp section
            {
                quickSortParallelDivideThreads(arr + i, lenArray - i, numThreads - numThreads / 2);
            }
        }
    }
}

void quicksortFullyParallelRecursive(int *arr, int left, int right) {
    int i = left;
    int j = right;
    int pivot = arr[left];

    // why does it not work when I use this ... ??
//#pragma omp task default(none) shared(i, j, arr) firstprivate(pivot)
    {
        while (i <= j) {
            while (arr[i] < pivot) {
                i++;
            }
            while (arr[j] > pivot) {
                j--;
            }
            if (i <= j) {
                int t = arr[i];
                arr[i] = arr[j];
                arr[j] = t;
                i++;
                j--;
            }
        }
    }

    if (right - left < 4096) {
        quicksortFullyParallelIntoSeq(arr, left, j);
        quicksortFullyParallelIntoSeq(arr, i, right);
    } else {
#pragma omp task default(none) firstprivate(arr, left, j)
        {
            if (left < j)
                quicksortFullyParallelRecursive(arr, left, j);
        }

#pragma omp task default(none) firstprivate (arr, right, i)
        {
            if (i < right)
                quicksortFullyParallelRecursive(arr, i, right);
        }
    }
}

void quicksortFullyParallel(int *arr, int len) {
#pragma omp parallel
    {
#pragma omp single nowait
        {
            quicksortFullyParallelRecursive(arr, 0, len - 1);
        }
    }
}

int main() {

    const auto threadCount = std::thread::hardware_concurrency();
    std::cout << "Using " << threadCount << " threads.\n";
    omp_set_num_threads(threadCount);
    wi::jobsystem::Initialize(threadCount);

    const auto generateRandomNumbers = [](size_t size) {
        std::vector<int> data(size);
        std::srand(std::time(nullptr));
        for (size_t i = 0; i < size; ++i) {
            data[i] = std::rand();
        }
        return data;
    };

    const auto testStdSort = [&](size_t size) -> void {
        auto testVector = generateRandomNumbers(size);
        {
            SCOPED_PROFILE_LOG("STD::SORT")
            std::sort(testVector.begin(), testVector.end());
        }
        if (std::is_sorted(testVector.begin(), testVector.end())) {
            std::cout << "Vector with " << size << " elements is sorted.\n" << std::endl;
        } else {
            std::cout << "Vector with " << size << " elements is NOT sorted!\n" << std::endl;
        }
    };

    const auto testStdParallelSort = [&](size_t size) -> void {
        auto testVector = generateRandomNumbers(size);
        {
            SCOPED_PROFILE_LOG("STD::SORT PARALLEL")
            std::sort(std::execution::par, testVector.begin(), testVector.end());
        }
        if (std::is_sorted(testVector.begin(), testVector.end())) {
            std::cout << "Vector with " << size << " elements is sorted.\n" << std::endl;
        } else {
            std::cout << "Vector with " << size << " elements is NOT sorted!\n" << std::endl;
        }
    };

    const auto testQuicksortSequential = [&](size_t size) -> void {
        auto testVector = generateRandomNumbers(size);
        {
            SCOPED_PROFILE_LOG("SEQUENTIAL")
            quickSortSequential(testVector, 0, testVector.size() - 1);
        }
        if (std::is_sorted(testVector.begin(), testVector.end())) {
            std::cout << "Vector with " << size << " elements is sorted.\n" << std::endl;
        } else {
            std::cout << "Vector with " << size << " elements is NOT sorted!\n" << std::endl;
        }
    };

    const auto testQuicksortParallel = [&](size_t size) -> void {
        auto testVector = generateRandomNumbers(size);
        {
            SCOPED_PROFILE_LOG("PARALLEL")
            quickSortParallel(testVector, 0, testVector.size() - 1);
        }
        if (std::is_sorted(testVector.begin(), testVector.end())) {
            std::cout << "Vector with " << size << " elements is sorted.\n" << std::endl;
        } else {
            std::cout << "Vector with " << size << " elements is NOT sorted!\n" << std::endl;
        }
    };

    const auto  testQuicksortFullyParallel = [&](size_t size) -> void {
        auto testVector = generateRandomNumbers(size);
        {
            SCOPED_PROFILE_LOG("FULLY PARALLEL")
            quicksortFullyParallel(&testVector[0], testVector.size());
        }
        if (std::is_sorted(testVector.begin(), testVector.end())) {
            std::cout << "Vector with " << size << " elements is sorted.\n" << std::endl;
        } else {
            std::cout << "Vector with " << size << " elements is NOT sorted!\n" << std::endl;
        }
    };

    const auto testQuicksortJob = [&](size_t size) -> void {
        auto testVector = generateRandomNumbers(size);
        {
            SCOPED_PROFILE_LOG("QS JOB")
            qsJob::qsFullyParallelRecursive(&testVector[0], 0, testVector.size() - 1, 0);
        }
        if (std::is_sorted(testVector.begin(), testVector.end())) {
            std::cout << "Vector with " << size << " elements is sorted.\n" << std::endl;
        } else {
            std::cout << "Vector with " << size << " elements is NOT sorted!\n" << std::endl;
        }
    };

    for (int i = 0; i < 8; i++) {
        auto sz = std::pow(10, i);
        testStdSort(sz);
        testStdParallelSort(sz);
        testQuicksortSequential(sz);
        testQuicksortParallel(sz);
        testQuicksortFullyParallel(sz);
        testQuicksortJob(sz);
    }

    return 0;
}