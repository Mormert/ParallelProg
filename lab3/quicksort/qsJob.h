

#ifndef PARALLELQUICKSORT_QSJOB_H
#define PARALLELQUICKSORT_QSJOB_H

#include "WickedEngine/wiJobSystem.h"

namespace qsJob {
    void qsFullyParallelIntoSeq(int *arr, int left, int right) {
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
            qsFullyParallelIntoSeq(arr, left, j);

        if (i < right)
            qsFullyParallelIntoSeq(arr, i, right);

    }

    void qsFullyParallelRecursive(int *arr, int left, int right, int depth) {
        int i = left;
        int j = right;
        int pivot = arr[left];
        depth += 1;

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

        // 32 threads, so
        // log2(32) = 5
        // we add 1 for balance.
        // thus when depth is larger than 6, we go sequential.
        if (right - left < 4096 || depth > 6) {
            qsFullyParallelIntoSeq(arr, left, j);
            qsFullyParallelIntoSeq(arr, i, right);
        } else {

            wi::jobsystem::context ctx;

            if (left < j) {
                wi::jobsystem::Execute(
                        ctx, [arr, left, j, &depth](wi::jobsystem::JobArgs args) {
                            qsFullyParallelRecursive(arr, left, j, depth);
                        });
            }

            {
                if (i < right) {
                    wi::jobsystem::Execute(
                            ctx, [arr, i, right, &depth](wi::jobsystem::JobArgs args) {
                                qsFullyParallelRecursive(arr, i, right, depth);
                            });
                }
            }

            Wait(ctx);

        }
    }
}


#endif //PARALLELQUICKSORT_QSJOB_H
