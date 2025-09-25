// CannonMPI_pt2pt.cpp
// Parallel matrix multiplication using Cannon's algorithm.
// Data distribution and collection use only point-to-point primitives (MPI_Send/MPI_Recv).
// Build: mpicxx -O2 -std=c++11 CannonMPI_pt2pt.cpp -o cannon_pt2pt
// Run example (4 processes): mpirun -np 4 ./cannon_pt2pt 8 8 8
// Command line args: M K N  (A is MxK, B is KxN -> C is MxN)
// NOTE: number of processes must be a perfect square and M,N must be divisible by sqrt(p).

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <chrono>

using namespace std;
using namespace std::chrono;

// allocate contiguous rows x cols matrix and return int** where row pointers point into a contiguous block.
// Free by freeMatrix(mat)
int allocMatrix(int*** mat, int rows, int cols) {
    int *data = (int*)malloc(sizeof(int) * rows * cols);
    if (!data) return -1;
    int **m = (int**)malloc(sizeof(int*) * rows);
    if (!m) {
        free(data);
        return -1;
    }
    for (int i = 0; i < rows; ++i) m[i] = data + i * cols;
    *mat = m;
    return 0;
}

int freeMatrix(int ***mat) {
    if (!mat || !(*mat)) return -1;
    free((*mat)[0]); // free contiguous block
    free(*mat);      // free row pointers
    *mat = NULL;
    return 0;
}

// compare two matrices equal element-wise
int compareMatrices(int **X, int **Y, int rows, int cols) {
    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j)
            if (X[i][j] != Y[i][j]) return 0;
    return 1;
}

void printMatrix(int **M, int rows, int cols) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) printf("%d ", M[i][j]);
        printf("\n");
    }
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int worldSize, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (argc < 4) {
        if (rank == 0) fprintf(stderr, "Usage: %s M K N\n", argv[0]);
        MPI_Finalize();
        return 1;
    }

    int M = atoi(argv[1]);
    int K = atoi(argv[2]);
    int N = atoi(argv[3]);

    // worldSize must be perfect square
    double sq = sqrt((double)worldSize);
    int p_sqrt = (int)sq;
    if (p_sqrt * p_sqrt != worldSize) {
        if (rank == 0) fprintf(stderr, "Number of processes must be a perfect square.\n");
        MPI_Finalize();
        return 1;
    }

    // block sizes must divide dims
    if (M % p_sqrt != 0 || N % p_sqrt != 0 || K % p_sqrt != 0) {
        if (rank == 0) {
            fprintf(stderr, "M, K, N must be divisible by sqrt(P) = %d\n", p_sqrt);
        }
        MPI_Finalize();
        return 1;
    }

    int procDim = p_sqrt;
    int blockM = M / procDim; // each process block rows in A and C
    int blockN = N / procDim; // each process block cols in B and C
    int blockK = K / procDim;

    // Create 2D Cartesian communicator (wrap-around)
    int dims[2] = {procDim, procDim};
    int periods[2] = {1,1}; // circular for Cannon
    int reorder = 1;
    MPI_Comm cartComm;
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, reorder, &cartComm);

    int coords[2];
    MPI_Cart_coords(cartComm, rank, 2, coords);
    int myRow = coords[0];
    int myCol = coords[1];

    // local blocks
    int **localA = NULL, **localB = NULL, **localC = NULL;
    allocMatrix(&localA, blockM, blockK);
    allocMatrix(&localB, blockK, blockN);
    allocMatrix(&localC, blockM, blockN);

    for (int i = 0; i < blockM; ++i)
        for (int j = 0; j < blockN; ++j)
            localC[i][j] = 0;

    // Rank 0 creates full matrices and sends blocks manually
    int **A = NULL, **B = NULL, **C = NULL;
    if (rank == 0) {
        allocMatrix(&A, M, K);
        allocMatrix(&B, K, N);
        allocMatrix(&C, M, N);

        // === Identity for A, random for B ===
        for (int i = 0; i < M; ++i)
            for (int j = 0; j < K; ++j)
                A[i][j] = (i == j ? 1 : 0);

        srand((unsigned)time(NULL));
        for (int i = 0; i < K; ++i)
            for (int j = 0; j < N; ++j)
                B[i][j] = rand() % 10;

        // Distribute blocks
        for (int r = 0; r < worldSize; ++r) {
            int rc[2];
            MPI_Cart_coords(cartComm, r, 2, rc);
            int rRow = rc[0], rCol = rc[1];

            int *tmpA = (int*)malloc(sizeof(int) * blockM * blockK);
            int *tmpB = (int*)malloc(sizeof(int) * blockK * blockN);
            for (int i = 0; i < blockM; ++i) {
                int global_i = rRow * blockM + i;
                for (int j = 0; j < blockK; ++j) {
                    int global_j = rCol * blockK + j;
                    tmpA[i * blockK + j] = A[global_i][global_j];
                }
            }
            for (int i = 0; i < blockK; ++i) {
                int global_i = rRow * blockK + i;
                for (int j = 0; j < blockN; ++j) {
                    int global_j = rCol * blockN + j;
                    tmpB[i * blockN + j] = B[global_i][global_j];
                }
            }

            if (r == 0) {
                for (int i = 0; i < blockM; ++i)
                    for (int j = 0; j < blockK; ++j)
                        localA[i][j] = tmpA[i * blockK + j];
                for (int i = 0; i < blockK; ++i)
                    for (int j = 0; j < blockN; ++j)
                        localB[i][j] = tmpB[i * blockN + j];
            } else {
                MPI_Send(tmpA, blockM * blockK, MPI_INT, r, 100 + r, MPI_COMM_WORLD);
                MPI_Send(tmpB, blockK * blockN, MPI_INT, r, 200 + r, MPI_COMM_WORLD);
            }
            free(tmpA);
            free(tmpB);
        }
    } else {
        MPI_Recv(&(localA[0][0]), blockM * blockK, MPI_INT, 0, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(&(localB[0][0]), blockK * blockN, MPI_INT, 0, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    // Start timing only after distribution
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = high_resolution_clock::now();

    // ---- Cannon algorithm ----
    int left, right, up, down;
    for (int s = 0; s < myRow; ++s) {
        MPI_Cart_shift(cartComm, 1, 1, &left, &right);
        MPI_Sendrecv_replace(&(localA[0][0]), blockM * blockK, MPI_INT, left, 11, right, 11, cartComm, MPI_STATUS_IGNORE);
    }
    for (int s = 0; s < myCol; ++s) {
        MPI_Cart_shift(cartComm, 0, 1, &up, &down);
        MPI_Sendrecv_replace(&(localB[0][0]), blockK * blockN, MPI_INT, up, 12, down, 12, cartComm, MPI_STATUS_IGNORE);
    }

    int **tempRes = NULL;
    allocMatrix(&tempRes, blockM, blockN);

    for (int step = 0; step < procDim; ++step) {
        for (int i = 0; i < blockM; ++i)
            for (int j = 0; j < blockN; ++j) tempRes[i][j] = 0;

        for (int i = 0; i < blockM; ++i)
            for (int k = 0; k < blockK; ++k) {
                int aij = localA[i][k];
                for (int j = 0; j < blockN; ++j)
                    tempRes[i][j] += aij * localB[k][j];
            }

        for (int i = 0; i < blockM; ++i)
            for (int j = 0; j < blockN; ++j)
                localC[i][j] += tempRes[i][j];

        MPI_Cart_shift(cartComm, 1, 1, &left, &right);
        MPI_Sendrecv_replace(&(localA[0][0]), blockM * blockK, MPI_INT, left, 21, right, 21, cartComm, MPI_STATUS_IGNORE);

        MPI_Cart_shift(cartComm, 0, 1, &up, &down);
        MPI_Sendrecv_replace(&(localB[0][0]), blockK * blockN, MPI_INT, up, 22, down, 22, cartComm, MPI_STATUS_IGNORE);
    }

    freeMatrix(&tempRes);

    // Gather results
    if (rank == 0) {
        for (int i = 0; i < blockM; ++i)
            for (int j = 0; j < blockN; ++j)
                C[i][j] = localC[i][j];

        for (int r = 1; r < worldSize; ++r) {
            int rc[2];
            MPI_Cart_coords(cartComm, r, 2, rc);
            int rRow = rc[0], rCol = rc[1];
            int *tmp = (int*)malloc(sizeof(int) * blockM * blockN);
            MPI_Recv(tmp, blockM * blockN, MPI_INT, r, 300 + r, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            for (int i = 0; i < blockM; ++i) {
                int global_i = rRow * blockM + i;
                for (int j = 0; j < blockN; ++j) {
                    int global_j = rCol * blockN + j;
                    C[global_i][global_j] = tmp[i * blockN + j];
                }
            }
            free(tmp);
        }
    } else {
        int *tmp = (int*)malloc(sizeof(int) * blockM * blockN);
        for (int i = 0; i < blockM; ++i)
            for (int j = 0; j < blockN; ++j)
                tmp[i * blockN + j] = localC[i][j];
        MPI_Send(tmp, blockM * blockN, MPI_INT, 0, 300 + rank, MPI_COMM_WORLD);
        free(tmp);
    }

    // Stop timing
    MPI_Barrier(MPI_COMM_WORLD);
    auto end = high_resolution_clock::now();

    if (rank == 0) {
        auto duration = duration_cast<milliseconds>(end - start);
        printf("Execution time: %lld ms\n", (long long)duration.count());

        // === Correctness: since A is I, expect C == B ===
        int ok = compareMatrices(B, C, M, N);
        if (ok) {
            printf("Result CORRECT (C == B, as A was identity).\n");
        } else {
            printf("Result INCORRECT! C != B\n");
            printf("B (first few rows):\n");
            printMatrix(B, (K<8?K:8), (N<8?N:8));
            printf("C (computed) (first few rows):\n");
            printMatrix(C, (M<8?M:8), (N<8?N:8));
        }
    }

    // cleanup
    freeMatrix(&localA);
    freeMatrix(&localB);
    freeMatrix(&localC);
    if (rank == 0) {
        freeMatrix(&A);
        freeMatrix(&B);
        freeMatrix(&C);
    }

    MPI_Comm_free(&cartComm);
    MPI_Finalize();
    return 0;
}
