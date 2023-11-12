#include <stdio.h>
#include <omp.h>

void foo() {
}

int main() {
    omp_set_num_threads(omp_get_num_procs());

    double start_time = omp_get_wtime();

#pragma omp parallel
{
    // Will run threaded in omp_get_num_procs amount of times.
    foo();
}

    printf("All threads completed in %f sec.\n", omp_get_wtime() - start_time);

    return 0;
}