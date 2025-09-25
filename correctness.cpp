#include <mpi.h>
#include <iostream>
#include <vector>
#include <cstdlib>
#include <cmath>
#include <algorithm>

// Serial matrix multiplication for correctness check
void serial_matrix_mult(const std::vector<double>& A, const std::vector<double>& B, std::vector<double>& C,
                        int M, int N, int K) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            double sum = 0.0;
            for (int k = 0; k < K; ++k) {
                sum += A[i*K + k] * B[k*N + j];
            }
            C[i*N + j] = sum;
        }
    }
}

// Compare two matrices element-wise
bool compare_matrices(const std::vector<double>& C1, const std::vector<double>& C2, int M, int N, double tol = 1e-6) {
    for (int i = 0; i < M*N; ++i) {
        if (std::abs(C1[i] - C2[i]) > tol) {
            return false;
        }
    }
    return true;
}

// Parallel multiplication for assigned rows
void multiply_rows(const std::vector<double>& A, const std::vector<double>& B, std::vector<double>& C,
                   int local_rows, int N, int K) {
    for (int i = 0; i < local_rows; ++i) {
        for (int j = 0; j < N; ++j) {
            double sum = 0.0;
            for (int k = 0; k < K; ++k) {
                sum += A[i*K + k] * B[k*N + j];
            }
            C[i*N + j] = sum;
        }
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int M = 8, K = 8, N = 8; // Example sizes

    std::vector<double> A, B, C;

    if (rank == 0) {
        A.resize(M*K);
        B.resize(K*N);
        C.resize(M*N, 0.0);
        for (int i = 0; i < M*K; ++i) A[i] = rand() % 10;
        for (int i = 0; i < K*N; ++i) B[i] = rand() % 10;
    }

    // Manual broadcast of B
    if (rank == 0) {
        for (int p = 1; p < size; ++p) {
            MPI_Send(B.data(), K*N, MPI_DOUBLE, p, 0, MPI_COMM_WORLD);
        }
    } else {
        B.resize(K*N);
        MPI_Recv(B.data(), K*N, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    // Divide rows of A
    int rows_per_proc = M / size;
    int extra = M % size;
    int row_start = rank * rows_per_proc + std::min(rank, extra);
    int row_end = row_start + rows_per_proc + (rank < extra ? 1 : 0);
    int local_rows = row_end - row_start;

    std::vector<double> local_A(local_rows * K);
    std::vector<double> local_C(local_rows * N, 0.0);

    // Distribute A
    if (rank == 0) {
        for (int p = 1; p < size; ++p) {
            int p_row_start = p * rows_per_proc + std::min(p, extra);
            int p_row_end = p_row_start + rows_per_proc + (p < extra ? 1 : 0);
            int p_rows = p_row_end - p_row_start;
            MPI_Send(A.data() + p_row_start*K, p_rows*K, MPI_DOUBLE, p, 1, MPI_COMM_WORLD);
        }
        std::copy(A.begin() + row_start*K, A.begin() + row_end*K, local_A.begin());
    } else {
        MPI_Recv(local_A.data(), local_rows*K, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    // Local multiplication
    multiply_rows(local_A, B, local_C, local_rows, N, K);

    // Gather results
    if (rank == 0) {
        std::copy(local_C.begin(), local_C.end(), C.begin() + row_start*N);
        for (int p = 1; p < size; ++p) {
            int p_row_start = p * rows_per_proc + std::min(p, extra);
            int p_row_end = p_row_start + rows_per_proc + (p < extra ? 1 : 0);
            int p_rows = p_row_end - p_row_start;
            MPI_Recv(C.data() + p_row_start*N, p_rows*N, MPI_DOUBLE, p, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    } else {
        MPI_Send(local_C.data(), local_rows*N, MPI_DOUBLE, 0, 2, MPI_COMM_WORLD);
    }

    // Correctness test (only on root)
    if (rank == 0) {
        std::vector<double> C_ref(M*N, 0.0);
        serial_matrix_mult(A, B, C_ref, M, N, K);
        if (compare_matrices(C, C_ref, M, N)) {
            std::cout << "Test PASSED: Parallel and serial results match." << std::endl;
        } else {
            std::cout << "Test FAILED: Results do not match." << std::endl;
        }
    }

    MPI_Finalize();
    return 0;
}
