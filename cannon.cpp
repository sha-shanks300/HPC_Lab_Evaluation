#include <mpi.h>
#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <iomanip>

using namespace std;

// Matrix element access in row-major order
#define IDX(i, j, ld) ((i) * (ld) + (j))

// Multiply blocks: C_block += A_block * B_block
void multiply_add_blocks(const vector<double>& A, const vector<double>& B, vector<double>& C,
                         int block_rows, int block_inner, int block_cols) {
    for (int i = 0; i < block_rows; ++i) {
        for (int j = 0; j < block_cols; ++j) {
            double sum = 0;
            for (int k = 0; k < block_inner; ++k) {
                sum += A[IDX(i, k, block_inner)] * B[IDX(k, j, block_cols)];
            }
            C[IDX(i, j, block_cols)] += sum;
        }
    }
}

// Serial matrix multiplication for correctness check
void serial_matrix_mult(const vector<double>& A, const vector<double>& B, vector<double>& C,
                        int M, int N, int K) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            double sum = 0.0;
            for (int k = 0; k < K; ++k) {
                sum += A[IDX(i, k, K)] * B[IDX(k, j, N)];
            }
            C[IDX(i, j, N)] = sum;
        }
    }
}

// Compare two matrices element-wise
bool compare_matrices(const vector<double>& C1, const vector<double>& C2, int M, int N, double tol = 1e-6) {
    for (int i = 0; i < M * N; ++i) {
        if (fabs(C1[i] - C2[i]) > tol) {
            return false;
        }
    }
    return true;
}

// Print a matrix (only used for small sizes)
void print_matrix(const vector<double>& mat, int rows, int cols, const string& name) {
    cout << name << ":\n";
    cout << fixed << setprecision(6);
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            cout << mat[IDX(i, j, cols)] << " ";
        }
        cout << "\n";
    }
    cout << flush;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size != 4) {
        if (rank == 0)
            cerr << "Error: This program requires exactly 4 MPI processes.\n";
        MPI_Finalize();
        return -1;
    }

    // Grid dimension
    const int q = 2;

    // Matrix sizes (must be divisible by q)
    const int M = 10000;
    const int K = 10000;
    const int N = 10000;

    if (M % q != 0 || K % q != 0 || N % q != 0) {
        if (rank == 0)
            cerr << "Error: M, K, N must be divisible by " << q << "\n";
        MPI_Finalize();
        return -1;
    }

    int block_M = M / q;
    int block_K = K / q;
    int block_N = N / q;

    int row = rank / q;
    int col = rank % q;

    // Initialize random seed differently for each rank
    srand(time(0) + rank * 1234);

    // Full matrices only on rank 0
    vector<double> A, B, C;
    if (rank == 0) {
        A.resize(M * K);
        B.resize(K * N);
        C.resize(M * N, 0);

        for (int i = 0; i < M * K; ++i)
            A[i] = (double)(rand() % 10); // integer values for easier debugging

        for (int i = 0; i < K * N; ++i)
            B[i] = (double)(rand() % 10);
    }

    // Local blocks for each process
    vector<double> A_block(block_M * block_K);
    vector<double> B_block(block_K * block_N);
    vector<double> C_block(block_M * block_N, 0);

    // Scatter A blocks manually
    if (rank == 0) {
        // Send A blocks to other ranks
        for (int r = 1; r < size; ++r) {
            int r_row = r / q;
            int r_col = r % q;
            vector<double> temp(block_M * block_K);
            for (int i = 0; i < block_M; ++i) {
                for (int j = 0; j < block_K; ++j) {
                    temp[IDX(i, j, block_K)] = A[IDX(r_row * block_M + i, r_col * block_K + j, K)];
                }
            }
            MPI_Send(temp.data(), block_M * block_K, MPI_DOUBLE, r, 0, MPI_COMM_WORLD);
        }
        // Copy own block
        for (int i = 0; i < block_M; ++i) {
            for (int j = 0; j < block_K; ++j) {
                A_block[IDX(i, j, block_K)] = A[IDX(0 * block_M + i, 0 * block_K + j, K)];
            }
        }
    } else {
        MPI_Recv(A_block.data(), block_M * block_K, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    // Scatter B blocks manually
    if (rank == 0) {
        // Send B blocks to other ranks
        for (int r = 1; r < size; ++r) {
            int r_row = r / q;
            int r_col = r % q;
            vector<double> temp(block_K * block_N);
            for (int i = 0; i < block_K; ++i) {
                for (int j = 0; j < block_N; ++j) {
                    temp[IDX(i, j, block_N)] = B[IDX(r_row * block_K + i, r_col * block_N + j, N)];
                }
            }
            MPI_Send(temp.data(), block_K * block_N, MPI_DOUBLE, r, 1, MPI_COMM_WORLD);
        }
        // Copy own block
        for (int i = 0; i < block_K; ++i) {
            for (int j = 0; j < block_N; ++j) {
                B_block[IDX(i, j, block_N)] = B[IDX(0 * block_K + i, 0 * block_N + j, N)];
            }
        }
    } else {
        MPI_Recv(B_block.data(), block_K * block_N, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    // Helper function for 2D rank grid
    auto rank_2d = [q](int r, int c) { return ((r + q) % q) * q + ((c + q) % q); };

    // Initial shifts (Cannon's algorithm)
    vector<double> A_temp = A_block;
    vector<double> B_temp = B_block;

    // A block shifts left by its row index
    int src_A = rank_2d(row, (col + row) % q);
    int dst_A = rank_2d(row, (col - row + q) % q);
    MPI_Sendrecv_replace(A_block.data(), block_M * block_K, MPI_DOUBLE,
                         dst_A, 10, src_A, 10, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    // B block shifts up by its column index
    int src_B = rank_2d((row + col) % q, col);
    int dst_B = rank_2d((row - col + q) % q, col);
    MPI_Sendrecv_replace(B_block.data(), block_K * block_N, MPI_DOUBLE,
                         dst_B, 20, src_B, 20, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    // Cannon steps
    for (int step = 0; step < q; ++step) {
        multiply_add_blocks(A_block, B_block, C_block, block_M, block_K, block_N);

        int dst_A_shift = rank_2d(row, (col - 1 + q) % q);
        int src_A_shift = rank_2d(row, (col + 1) % q);
        MPI_Sendrecv_replace(A_block.data(), block_M * block_K, MPI_DOUBLE,
                             dst_A_shift, 30, src_A_shift, 30, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        int dst_B_shift = rank_2d((row - 1 + q) % q, col);
        int src_B_shift = rank_2d((row + 1) % q, col);
        MPI_Sendrecv_replace(B_block.data(), block_K * block_N, MPI_DOUBLE,
                             dst_B_shift, 40, src_B_shift, 40, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    // Gather C blocks to rank 0 manually
    if (rank == 0) {
        // Copy own block
        for (int i = 0; i < block_M; ++i) {
            for (int j = 0; j < block_N; ++j) {
                C[IDX(i, j, N)] = C_block[IDX(i, j, block_N)];
            }
        }

        for (int r = 1; r < size; ++r) {
            int r_row = r / q;
            int r_col = r % q;
            vector<double> temp(block_M * block_N);
            MPI_Recv(temp.data(), block_M * block_N, MPI_DOUBLE, r, 50, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            for (int i = 0; i < block_M; ++i) {
                for (int j = 0; j < block_N; ++j) {
                    C[IDX(r_row * block_M + i, r_col * block_N + j, N)] = temp[IDX(i, j, block_N)];
                }
            }
        }
    } else {
        MPI_Send(C_block.data(), block_M * block_N, MPI_DOUBLE, 0, 50, MPI_COMM_WORLD);
    }

    // Correctness check on rank 0
    if (rank == 0) {
        vector<double> C_ref(M * N, 0);
        serial_matrix_mult(A, B, C_ref, M, N, K);

        // NOTE: The print_matrix function is commented out to avoid excessive output for large matrices.
        // Uncomment for small test cases (e.g., M=4, N=4, K=4)
        /*
        cout << "\nMatrix A:\n"; print_matrix(A, M, K, "A");
        cout << "\nMatrix B:\n"; print_matrix(B, K, N, "B");
        cout << "\nResult Matrix C (Parallel):\n"; print_matrix(C, M, N, "C_parallel");
        cout << "\nResult Matrix C (Serial):\n"; print_matrix(C_ref, M, N, "C_serial");
        */

        if (compare_matrices(C, C_ref, M, N)) {
            cout << "\nTest PASSED: Parallel and serial results match.\n";
        } else {
            cout << "\nTest FAILED: Results do not match.\n";
        }
    }

    MPI_Finalize();
    return 0;
}