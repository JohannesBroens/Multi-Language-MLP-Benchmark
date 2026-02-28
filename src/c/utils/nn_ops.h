#ifndef NN_OPS_H
#define NN_OPS_H

#ifdef __cplusplus
extern "C" {
/* CUDA/C++ does not support C99 restrict -- use __restrict__ instead */
#define restrict __restrict__
#endif

/* --- Optimizer and scheduler selection --- */
typedef enum { OPT_SGD = 0, OPT_ADAM = 1 } OptimizerType;
typedef enum { SCHED_NONE = 0, SCHED_COSINE = 1 } SchedulerType;

/*
 * Shared neural network operations used across all model implementations.
 * Provides GEMM (matrix multiply), activation functions, loss computation,
 * and SGD parameter updates.
 *
 * Design rationale: extracting these from individual models avoids code
 * duplication and ensures that GEMM optimizations (cache tiling, OpenMP
 * parallelism thresholds) are consistent across MLP, CNN, and future models.
 * This follows the PyTorch nn.functional pattern -- shared ops, swappable models.
 */

/* Cache tile size: 3 tiles of 64x64 floats ~ 48KB, fits L1 cache.
 * Chosen empirically on Intel/AMD desktop CPUs with 32-48KB L1d. */
#define NN_TILE 64

/* Minimum FLOPs (M*N*K) before spawning OpenMP threads.
 * Thread overhead ~1-5us per parallel region (libgomp persistent team).
 * At ~4 GFLOP/s single-core, 100K FLOPs = ~25us work per thread. */
#define NN_FLOP_THRESHOLD 100000L

/* Minimum elements for elementwise OpenMP (bias, relu, softmax, SGD).
 * Each element ~2-5 FLOPs; 200K elements = ~50-125us per thread. */
#define NN_ELEM_THRESHOLD 200000L

/* --- GEMM: C = A @ B variants for row-major float matrices --- */

/* C[M,N] = A[M,K] @ B[K,N]  (both normal) */
void nn_sgemm_nn(int M, int N, int K,
                 const float *restrict A,
                 const float *restrict B,
                 float *restrict C);

/* C[M,N] = A^T[M,K] @ B[K,N]  (A stored as [K,M]) */
void nn_sgemm_tn(int M, int N, int K,
                 const float *restrict A,
                 const float *restrict B,
                 float *restrict C);

/* C[M,N] = A[M,K] @ B^T[K,N]  (B stored as [N,K]) */
void nn_sgemm_nt(int M, int N, int K,
                 const float *restrict A,
                 const float *restrict B,
                 float *restrict C);

/* --- Activations --- */

/* In-place ReLU: x[i] = max(0, x[i]) */
void nn_relu_forward(float *x, int n);

/* In-place ReLU backward: grad[i] = (x[i] > 0) ? grad[i] : 0 */
void nn_relu_backward(float *grad, const float *x, int n);

/* In-place bias + ReLU fused: x[i*width+j] += bias[j], then ReLU */
void nn_bias_relu(float *x, const float *bias, int rows, int width);

/* --- Softmax + Cross-Entropy --- */

/* In-place numerically stable softmax per row.
 * x is [rows, cols], each row independently softmaxed. */
void nn_softmax(float *x, int rows, int cols);

/* In-place bias + softmax fused */
void nn_bias_softmax(float *x, const float *bias, int rows, int cols);

/* Cross-entropy loss: returns average -log(prob[target]) over rows */
float nn_cross_entropy_loss(const float *probs, const int *targets, int rows, int cols);

/* Cross-entropy gradient: grad = probs - one_hot(targets).
 * Also computes per-sample losses if losses != NULL. */
void nn_cross_entropy_grad(const float *probs, const int *targets,
                           float *grad, float *losses, int rows, int cols);

/* --- Column sum: out[j] = sum_i(mat[i*cols+j]) --- */
void nn_col_sum(const float *mat, float *out, int rows, int cols);

/* --- SGD update: param[i] -= lr * grad[i] --- */
void nn_sgd_update(float *param, const float *grad, float lr, int n);

/* --- PRNG for weight initialization --- */

/* Xorshift32 PRNG (matches across C and Rust implementations) */
unsigned int nn_xorshift32(unsigned int *state);

/* Random float in [0, 1) */
float nn_rand_float(unsigned int *state);

/* Xavier uniform initialization: weights ~ Uniform(-scale, scale)
 * where scale = sqrt(2 / fan_in) */
void nn_xavier_init(float *weights, int size, int fan_in, unsigned int *rng);

/* --- Physical core detection (Linux) --- */

/* Returns number of physical (non-HT) CPU cores.
 * Used to size OpenMP thread team -- HT cores share FPU,
 * causing contention on compute-bound GEMM. */
int nn_detect_physical_cores(void);

#ifdef __cplusplus
}
#endif

#endif /* NN_OPS_H */
