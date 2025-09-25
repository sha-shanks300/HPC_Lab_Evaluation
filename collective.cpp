
#include <mpi.h>
#include <iostream>
#include <vector>
#include <cstdlib>
#include <algorithm>
#include <iomanip>

// Multiply rows of A with B and store in C
void multiply_rows(const std::vector<double>& A, const std::vector<double>& B,
std::vector<double>& C, int local_rows, int N, int K) {
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

int M = 4000, K = 4000, N = 4000; // Adjust matrix size here

std::vector<double> A, B, C;
if (rank == 0) {
// Initialize matrices with random values
srand(42); // For reproducibility
A.resize(M*K);
B.resize(K*N);
C.resize(M*N, 0.0);
for (int i = 0; i < M*K; ++i) A[i] = rand() % 10;
for (int i = 0; i < K*N; ++i) B[i] = rand() % 10;
}

// ========== POINT-TO-POINT COMMUNICATION ==========
double start_ptp = MPI_Wtime();

// Send B to all processes manually
if (rank == 0) {
for (int p = 1; p < size; ++p)
MPI_Send(B.data(), K*N, MPI_DOUBLE, p, 0, MPI_COMM_WORLD);
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

std::vector<double> local_A(local_rows*K);
std::vector<double> local_C(local_rows*N, 0.0);

// Distribute A manually
if (rank == 0) {
for (int p = 1; p < size; ++p) {
int p_start = p * rows_per_proc + std::min(p, extra);
int p_end = p_start + rows_per_proc + (p < extra ? 1 : 0);
MPI_Send(A.data() + p_start*K, (p_end - p_start)*K, MPI_DOUBLE, p, 1, MPI_COMM_WORLD);
}
std::copy(A.begin() + row_start*K, A.begin() + row_end*K, local_A.begin());
} else {
MPI_Recv(local_A.data(), local_rows*K, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
}

multiply_rows(local_A, B, local_C, local_rows, N, K);

// Gather results manually
if (rank == 0) {
std::copy(local_C.begin(), local_C.end(), C.begin() + row_start*N);
for (int p = 1; p < size; ++p) {
int p_start = p * rows_per_proc + std::min(p, extra);
int p_end = p_start + rows_per_proc + (p < extra ? 1 : 0);
MPI_Recv(C.data() + p_start*N, (p_end - p_start)*N, MPI_DOUBLE, p, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
}
} else {
MPI_Send(local_C.data(), local_rows*N, MPI_DOUBLE, 0, 2, MPI_COMM_WORLD);
}

double end_ptp = MPI_Wtime();

// ========== COLLECTIVE COMMUNICATION ==========
MPI_Bcast(B.data(), K*N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

// Prepare scatter info
std::vector<int> sendcounts(size), displs(size);
for (int i = 0; i < size; ++i) {
int start = i * rows_per_proc + std::min(i, extra);
int end = start + rows_per_proc + (i < extra ? 1 : 0);
sendcounts[i] = (end - start)*K;
displs[i] = start*K;
}

int local_rows_coll = sendcounts[rank] / K;
std::vector<double> local_A_coll(sendcounts[rank]);
std::vector<double> local_C_coll(local_rows_coll*N, 0.0);

double start_coll = MPI_Wtime();

MPI_Scatterv(A.data(), sendcounts.data(), displs.data(), MPI_DOUBLE,
local_A_coll.data(), sendcounts[rank], MPI_DOUBLE, 0, MPI_COMM_WORLD);

multiply_rows(local_A_coll, B, local_C_coll, local_rows_coll, N, K);

// Gather results collectively
std::vector<int> recvcounts(size), recvdispls(size);
for (int i = 0; i < size; ++i) {
int start = i * rows_per_proc + std::min(i, extra);
int end = start + rows_per_proc + (i < extra ? 1 : 0);
recvcounts[i] = (end - start)*N;
recvdispls[i] = start*N;
}

MPI_Gatherv(local_C_coll.data(), local_rows_coll*N, MPI_DOUBLE,
C.data(), recvcounts.data(), recvdispls.data(),
MPI_DOUBLE, 0, MPI_COMM_WORLD);

double end_coll = MPI_Wtime();

// ========== PRINT RESULTS ==========
if (rank == 0) {
std::cout << std::fixed << std::setprecision(6);
std::cout << "Matrix Size: " << M << " x " << N << "\n";
std::cout << "Point-to-Point Time: " << end_ptp - start_ptp << " s\n";
std::cout << "Collective Communication Time: " << end_coll - start_coll << " s\n";
std::cout << "Speedup: " << (end_ptp - start_ptp)/(end_coll - start_coll) << "\n";
}

MPI_Finalize();
return 0;
}
