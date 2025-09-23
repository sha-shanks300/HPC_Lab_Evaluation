You are tasked with implementing a parallel matrix-matrix multiplication, C = A x B, where A is an M x K matrix and B is a K x N matrix. 
The resulting matrix C will have dimensions M x N. Your implementation must use the Message Passing Interface (MPI) You are forbid- den from 
using any collective communication operations (€.8., MPI Scatter, MPI Gather, MPI_Altoall) for distributing the data and collecting the final results.
You must manually im- plement all communication logic using only point-to-point primitives (MPI Send, MPI Recv MPI Isend MPI Irecv). 
You will be evaluated not just on correctness but on the efficiency of your communication pattern.
1. Host your MPI code on GitHub, using commits to document changes
2. Create a C-++ function to test your code's correctness. Explain how this function works and why it guarantees accurate results
3. Analyze how your code's performance scales with increasing matrix size. Justify your findings using a results table
4. Explain how using collective communication operations would change your code's effi- ciency and complexity compared to your current point-to-point approach.
