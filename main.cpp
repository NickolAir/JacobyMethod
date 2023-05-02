#include <iostream>
#include <mpi.h>
#include <cmath>

#define X0 -1
#define X1 1
#define Y0 -1
#define Y1 1
#define Z0 -1
#define Z1 1

//Главный параметр уравнения
#define A 1e5

//Предел сходимости
#define E 1e-8

//Изачальное приближение
#define FI0 0

//Размеры сетки
#define NX 320
#define NY 320
#define NZ 320

//Шаги сетки
#define hx ((double)(X1-X0)/(NX-1))
#define hy ((double)(Y1-Y0)/(NY-1))
#define hz ((double)(Z1-Z0)/(NZ-1))

#define I(i, j, k) ((i)*NY*NZ+(j)*NZ+(k))

double fi(double i, double j, double k) {
    double x = X0 + (i) * hx;
    double y = X0 + (j) * hy;
    double z = X0 + (k) * hz;
    return x * x + y * y + z * z;
}

double ro(double i, double j, double k) {
    return 6 - A * fi(i, j, k);
}

bool isBoundaryElement(int index, int bound) {
    return index == 0 || index == bound - 1;
}

// инициализирует сетку и предыдущую сетку заданными значениями
void gridInit(double *grid, double *previous_grid, int pr_cells_count, int pr_cells_shift) {
    for (int i = 1; i < pr_cells_count + 1; i++) {
        for (int j = 0; j < NY; j++) {
            for (int k = 0; k < NZ; k++) {
                grid[I(i, j, k)] = FI0;
                previous_grid[I(i, j, k)] = FI0;

                int actual_i = i + pr_cells_shift;

                if (isBoundaryElement(actual_i, NX) || isBoundaryElement(j, NY) || isBoundaryElement(k, NZ)) {
                    grid[I(i, j, k)] = fi(actual_i, j, k);
                    previous_grid[I(i, j, k)] = fi(actual_i, j, k);
                }
            }
        }
    }
}

double iterationFunc(int i, int j, int k, double *grid, int pr_cells_shift) {
    return ((1.0 / (2.0 / (hx * hx) + 2.0 / (hy * hy) + 2.0 / (hz * hz) + A)) *
            ((grid[I(i + 1, j, k)] + grid[I(i - 1, j, k)]) / (hx * hx) +
             (grid[I(i, j + 1, k)] + grid[I(i, j - 1, k)]) / (hy * hy) +
             (grid[I(i, j, k + 1)] + grid[I(i, j, k - 1)]) / (hz * hz) -
             ro(i + pr_cells_shift, j, k)));
}

// обновляет значения текущей сетки и предыдущей сетки для заданного индекса
void updateGrid(int index, double *current_value, double *prev_value) {
    for (int j = 1; j < NY - 1; j++) {
        for (int k = 1; k < NZ - 1; k++) {
            prev_value[I(index, j, k)] = current_value[I(index, j, k)];
        }
    }
}

// обновляет значение ячейки сетки для заданного индекса и вычисляет максимальное изменение в этой ячейке
void updateGridCell(int index, double *grid, double *previous_grid, int pr_cells_shift, double *pr_diff) {
    int actual_index = index + pr_cells_shift;
    if (!isBoundaryElement(actual_index, NX)) {
        for (int j = 1; j < NY - 1; j++) {
            for (int k = 1; k < NZ - 1; k++) {
                grid[I(index, j, k)] = iterationFunc(index, j, k, previous_grid, pr_cells_shift);

                double cur_diff = fabs(grid[I(index, j, k)] - previous_grid[I(index, j, k)]);

                if (*pr_diff < cur_diff) {
                    *pr_diff = cur_diff;
                }
            }
        }
    }
    updateGrid(index, grid, previous_grid);
}

// обновляет значения ячеек сетки в центре сетки
void updateGridCenter(double *grid, double *previous_grid, int pr_cells_count, int pr_cells_shift, double *pr_diff) {
    int center = (pr_cells_count + 1) / 2;

    updateGridCell(center, grid, previous_grid, pr_cells_shift, pr_diff);

    for (int j = 1; j < (pr_cells_count + 1) / 2; j++) {
        updateGridCell(center - j, grid, previous_grid, pr_cells_shift, pr_diff);
        updateGridCell(center + j, grid, previous_grid, pr_cells_shift, pr_diff);
    }

    if (pr_cells_count % 2 == 0) {
        updateGridCell(pr_cells_count, grid, previous_grid, pr_cells_shift, pr_diff);
    }
}

// обновляет значения граничных элементов сетки
void updateGridBound(double *grid, double *previous_grid, int pr_cells_count, int pr_cells_shift, double *pr_diff) {
    updateGridCell(1, grid, previous_grid, pr_cells_shift, pr_diff);
    updateGridCell(pr_cells_count, grid, previous_grid, pr_cells_shift, pr_diff);
}

// отправка граничных элементов текущей сетки сетки на сторону соседнего процесса по указанной координате
void sendGridBound(double *grid, int pr_rank, int comm_size, int pr_cells_count, MPI_Request *request_prev,
                   MPI_Request *request_next) {
    int layer_size = NY * NZ;
    if (pr_rank != 0) {
        MPI_Isend(grid + layer_size, layer_size, MPI_DOUBLE, pr_rank - 1, 0, MPI_COMM_WORLD, &request_next[0]);
        MPI_Irecv(grid, layer_size, MPI_DOUBLE, pr_rank - 1, 0, MPI_COMM_WORLD, &request_next[1]);
    }
    if (pr_rank != comm_size - 1) {
        MPI_Isend(grid + (pr_cells_count) * layer_size, layer_size, MPI_DOUBLE, pr_rank + 1, 0, MPI_COMM_WORLD,
                  &request_prev[0]);
        MPI_Irecv(grid + (pr_cells_count + 1) * layer_size, layer_size, MPI_DOUBLE, pr_rank + 1, 0, MPI_COMM_WORLD,
                  &request_prev[1]);
    }
}

// прием граничных элементов соседнего процесса по указанной координате
void recieveGridBound(int pr_rank, int comm_size, MPI_Request *request_prev, MPI_Request *request_next) {
    if (pr_rank != 0) {
        MPI_Wait(&request_next[0], MPI_STATUS_IGNORE);
        MPI_Wait(&request_next[1], MPI_STATUS_IGNORE);
    }
    if (pr_rank != comm_size - 1) {
        MPI_Wait(&request_prev[0], MPI_STATUS_IGNORE);
        MPI_Wait(&request_prev[1], MPI_STATUS_IGNORE);
    }
}

int jacobyMethod(double *grid, int pr_cells_count, int pr_cells_shift) {
    double *previous_grid = (double *) malloc(sizeof(double) * (pr_cells_count + 2) * NY * NZ);
    gridInit(grid, previous_grid, pr_cells_count, pr_cells_shift);

    int iteration_count = 0;
    double diff = 0;

    int pr_rank, comm_size;
    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &pr_rank);

    MPI_Request request_prev[2];
    MPI_Request request_next[2];

    do {
        double pr_diff = 0;

        sendGridBound(previous_grid, pr_rank, comm_size, pr_cells_count, request_prev, request_next);
        updateGridCenter(grid, previous_grid, pr_cells_count, pr_cells_shift, &pr_diff);
        recieveGridBound(pr_rank, comm_size, request_prev, request_next);
        updateGridBound(grid, previous_grid, pr_cells_count, pr_cells_shift, &pr_diff);
        MPI_Allreduce(&pr_diff, &diff, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        iteration_count++;

    } while (diff >= E);

    free(previous_grid);

    return iteration_count;
}

int main(int argc, char *argv[]) {
    int pr_rank, comm_size;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &pr_rank);

    if (NX % comm_size != 0) {
        std::cerr << ("Grid size should divide of number of processes\n");
        MPI_Finalize();
        return 0;
    }

    int pr_cells_count = (NX / comm_size);
    int pr_cells_shift = pr_rank * pr_cells_count - 1;

    double *grid = (double *) malloc(sizeof(double) * (pr_cells_count + 2) * NY * NZ);

    double time_start = MPI_Wtime();
    int iteration_count = jacobyMethod(grid, pr_cells_count, pr_cells_shift);
    double time_end = MPI_Wtime();

    if (pr_rank == 0) {
        printf("Iteration count: %d\n", iteration_count);
        printf("Time taken: %f\n", time_end - time_start);
    }

    free(grid);
    MPI_Finalize();
    return 0;
}