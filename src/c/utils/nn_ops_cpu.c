#include "nn_ops.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <stdint.h>

#ifdef _OPENMP
#include <omp.h>
#include <unistd.h>

/* Detect physical core count for optimal OMP threads.
 * Hyperthreaded cores share FPU, so for compute-bound GEMM
 * using only physical cores avoids contention.
 * Falls back to omp_get_max_threads() if detection fails. */
int nn_detect_physical_cores(void) {
    int max_cpus = (int)sysconf(_SC_NPROCESSORS_ONLN);
    if (max_cpus <= 0) max_cpus = 1;

    int seen[256] = {0};
    int n_physical = 0;
    for (int cpu = 0; cpu < max_cpus && cpu < 256; ++cpu) {
        char path[128];
        snprintf(path, sizeof(path),
                 "/sys/devices/system/cpu/cpu%d/topology/core_id", cpu);
        FILE *f = fopen(path, "r");
        if (!f) { n_physical = 0; break; }
        int core_id = -1;
        if (fscanf(f, "%d", &core_id) == 1 && core_id >= 0 && core_id < 256) {
            if (!seen[core_id]) {
                seen[core_id] = 1;
                n_physical++;
            }
        }
        fclose(f);
    }
    return n_physical > 0 ? n_physical : max_cpus;
}

__attribute__((constructor))
static void init_omp_threads(void) {
    if (!getenv("OMP_NUM_THREADS")) {
        int cores = nn_detect_physical_cores();
        omp_set_num_threads(cores);
    }
}

#else

int nn_detect_physical_cores(void) {
    return 1;
}

#endif

#ifdef USE_BLAS
#include <cblas.h>
#endif

/* ------------------------------------------------------------------ */
/*  GEMM: cblas when available, cache-tiled OpenMP fallback           */
/* ------------------------------------------------------------------ */

#ifndef USE_BLAS

void nn_sgemm_nn(int M, int N, int K,
                 const float * restrict A,
                 const float * restrict B,
                 float * restrict C) {
    long flops = (long)M * N * K;

    if (K <= NN_TILE && N <= NN_TILE) {
        #pragma omp parallel for schedule(static) if(flops > NN_FLOP_THRESHOLD)
        for (int i = 0; i < M; ++i) {
            float * restrict Ci = C + i * N;
            const float * restrict Ai = A + i * K;
            for (int j = 0; j < N; ++j) Ci[j] = 0.0f;
            for (int k = 0; k < K; ++k) {
                float a = Ai[k];
                const float * restrict Bk = B + k * N;
                for (int j = 0; j < N; ++j)
                    Ci[j] += a * Bk[j];
            }
        }
    } else {
        memset(C, 0, (size_t)M * N * sizeof(float));
        #pragma omp parallel for schedule(static) if(flops > NN_FLOP_THRESHOLD)
        for (int i0 = 0; i0 < M; i0 += NN_TILE) {
            int imax = (i0 + NN_TILE < M) ? i0 + NN_TILE : M;
            for (int k0 = 0; k0 < K; k0 += NN_TILE) {
                int kmax = (k0 + NN_TILE < K) ? k0 + NN_TILE : K;
                for (int j0 = 0; j0 < N; j0 += NN_TILE) {
                    int jmax = (j0 + NN_TILE < N) ? j0 + NN_TILE : N;
                    for (int i = i0; i < imax; ++i) {
                        float * restrict Ci = C + i * N;
                        const float * restrict Ai = A + i * K;
                        for (int k = k0; k < kmax; ++k) {
                            float a_ik = Ai[k];
                            const float * restrict Bk = B + k * N;
                            for (int j = j0; j < jmax; ++j)
                                Ci[j] += a_ik * Bk[j];
                        }
                    }
                }
            }
        }
    }
}

void nn_sgemm_tn(int M, int N, int K,
                 const float * restrict A,
                 const float * restrict B,
                 float * restrict C) {
    long flops = (long)M * N * K;
    #pragma omp parallel for schedule(static) if(flops > NN_FLOP_THRESHOLD)
    for (int i = 0; i < M; ++i) {
        float * restrict Ci = C + i * N;
        for (int j = 0; j < N; ++j) Ci[j] = 0.0f;
        for (int k = 0; k < K; ++k) {
            float a_ki = A[k * M + i];
            const float * restrict Bk = B + k * N;
            for (int j = 0; j < N; ++j)
                Ci[j] += a_ki * Bk[j];
        }
    }
}

void nn_sgemm_nt(int M, int N, int K,
                 const float * restrict A,
                 const float * restrict B,
                 float * restrict C) {
    long flops = (long)M * N * K;
    #pragma omp parallel for schedule(static) if(flops > NN_FLOP_THRESHOLD)
    for (int i = 0; i < M; ++i) {
        const float * restrict Ai = A + i * K;
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            const float * restrict Bj = B + j * K;
            for (int k = 0; k < K; ++k)
                sum += Ai[k] * Bj[k];
            C[i * N + j] = sum;
        }
    }
}

#else /* USE_BLAS */

void nn_sgemm_nn(int M, int N, int K,
                 const float *A, const float *B, float *C) {
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                M, N, K, 1.0f, A, K, B, N, 0.0f, C, N);
}

void nn_sgemm_tn(int M, int N, int K,
                 const float *A, const float *B, float *C) {
    cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                M, N, K, 1.0f, A, M, B, N, 0.0f, C, N);
}

void nn_sgemm_nt(int M, int N, int K,
                 const float *A, const float *B, float *C) {
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                M, N, K, 1.0f, A, K, B, K, 0.0f, C, N);
}

#endif /* USE_BLAS */

/* ------------------------------------------------------------------ */
/*  Activations                                                        */
/* ------------------------------------------------------------------ */

void nn_relu_forward(float *x, int n) {
    #pragma omp parallel for schedule(static) if(n > NN_ELEM_THRESHOLD)
    for (int i = 0; i < n; ++i)
        if (x[i] < 0.0f) x[i] = 0.0f;
}

void nn_relu_backward(float *grad, const float *x, int n) {
    #pragma omp parallel for schedule(static) if(n > NN_ELEM_THRESHOLD)
    for (int i = 0; i < n; ++i)
        if (x[i] <= 0.0f) grad[i] = 0.0f;
}

void nn_bias_relu(float *x, const float *bias, int rows, int width) {
    long n = (long)rows * width;
    #pragma omp parallel for schedule(static) if(n > NN_ELEM_THRESHOLD)
    for (int i = 0; i < rows; ++i) {
        float *xi = x + i * width;
        for (int j = 0; j < width; ++j) {
            xi[j] += bias[j];
            if (xi[j] < 0.0f) xi[j] = 0.0f;
        }
    }
}

/* ------------------------------------------------------------------ */
/*  Softmax + Cross-Entropy                                            */
/* ------------------------------------------------------------------ */

void nn_softmax(float *x, int rows, int cols) {
    long n = (long)rows * cols;
    #pragma omp parallel for schedule(static) if(n > NN_ELEM_THRESHOLD)
    for (int i = 0; i < rows; ++i) {
        float *xi = x + i * cols;
        float max_val = -1e30f;
        for (int j = 0; j < cols; ++j)
            if (xi[j] > max_val) max_val = xi[j];
        float sum = 0.0f;
        for (int j = 0; j < cols; ++j) {
            xi[j] = expf(xi[j] - max_val);
            sum += xi[j];
        }
        for (int j = 0; j < cols; ++j)
            xi[j] /= sum;
    }
}

void nn_bias_softmax(float *x, const float *bias, int rows, int cols) {
    long n = (long)rows * cols;
    #pragma omp parallel for schedule(static) if(n > NN_ELEM_THRESHOLD)
    for (int i = 0; i < rows; ++i) {
        float *xi = x + i * cols;
        float max_val = -1e30f;
        for (int j = 0; j < cols; ++j) {
            xi[j] += bias[j];
            if (xi[j] > max_val) max_val = xi[j];
        }
        float sum = 0.0f;
        for (int j = 0; j < cols; ++j) {
            xi[j] = expf(xi[j] - max_val);
            sum += xi[j];
        }
        for (int j = 0; j < cols; ++j)
            xi[j] /= sum;
    }
}

float nn_cross_entropy_loss(const float *probs, const int *targets, int rows, int cols) {
    float total = 0.0f;
    #pragma omp parallel for reduction(+:total) schedule(static) if(rows > 10000)
    for (int i = 0; i < rows; ++i)
        total -= logf(probs[i * cols + targets[i]] + 1e-7f);
    return total / rows;
}

void nn_cross_entropy_grad(const float *probs, const int *targets,
                           float *grad, float *losses, int rows, int cols) {
    long n = (long)rows * cols;
    #pragma omp parallel for schedule(static) if(n > NN_ELEM_THRESHOLD)
    for (int i = 0; i < rows; ++i) {
        const float *p = probs + i * cols;
        float *g = grad + i * cols;
        if (losses)
            losses[i] = -logf(p[targets[i]] + 1e-7f);
        for (int j = 0; j < cols; ++j)
            g[j] = p[j] - ((targets[i] == j) ? 1.0f : 0.0f);
    }
}

/* ------------------------------------------------------------------ */
/*  Column sum                                                         */
/* ------------------------------------------------------------------ */

void nn_col_sum(const float *mat, float *out, int rows, int cols) {
    memset(out, 0, cols * sizeof(float));
    for (int i = 0; i < rows; ++i) {
        const float *row = mat + i * cols;
        for (int j = 0; j < cols; ++j)
            out[j] += row[j];
    }
}

/* ------------------------------------------------------------------ */
/*  Learning rate scheduler                                            */
/* ------------------------------------------------------------------ */

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

float nn_cosine_lr(int epoch, int total_epochs, float lr_max,
                   int warmup_epochs, float lr_min) {
    if (epoch < warmup_epochs) {
        return lr_max * ((float)(epoch + 1) / (float)warmup_epochs);
    }
    float progress = (float)(epoch - warmup_epochs) / (float)(total_epochs - warmup_epochs);
    return lr_min + 0.5f * (lr_max - lr_min) * (1.0f + cosf((float)M_PI * progress));
}

/* ------------------------------------------------------------------ */
/*  SGD update                                                         */
/* ------------------------------------------------------------------ */

void nn_sgd_update(float *param, const float *grad, float lr, int n) {
    #pragma omp parallel for schedule(static) if(n > NN_ELEM_THRESHOLD)
    for (int i = 0; i < n; ++i)
        param[i] -= lr * grad[i];
}

/* ------------------------------------------------------------------ */
/*  Adam optimizer                                                     */
/* ------------------------------------------------------------------ */

void nn_adam_state_init(AdamState *state, int n) {
    state->n = n;
    state->m = (float *)calloc(n, sizeof(float));
    state->v = (float *)calloc(n, sizeof(float));
}

void nn_adam_state_free(AdamState *state) {
    free(state->m); free(state->v);
    state->m = NULL; state->v = NULL;
}

void nn_adam_update(float *param, const float *grad, AdamState *state,
                    float lr, float beta1, float beta2, float eps, int t) {
    float bc1 = 1.0f - powf(beta1, (float)t);
    float bc2 = 1.0f - powf(beta2, (float)t);
    #pragma omp parallel for schedule(static) if(state->n > NN_ELEM_THRESHOLD)
    for (int i = 0; i < state->n; i++) {
        state->m[i] = beta1 * state->m[i] + (1.0f - beta1) * grad[i];
        state->v[i] = beta2 * state->v[i] + (1.0f - beta2) * grad[i] * grad[i];
        float m_hat = state->m[i] / bc1;
        float v_hat = state->v[i] / bc2;
        param[i] -= lr * m_hat / (sqrtf(v_hat) + eps);
    }
}

/* ------------------------------------------------------------------ */
/*  PRNG for weight initialization                                     */
/* ------------------------------------------------------------------ */

unsigned int nn_xorshift32(unsigned int *state) {
    unsigned int x = *state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *state = x;
    return x;
}

float nn_rand_float(unsigned int *state) {
    return (float)(nn_xorshift32(state) & 0x7FFFFFFF) / (float)0x7FFFFFFF;
}

void nn_xavier_init(float *weights, int size, int fan_in, unsigned int *rng) {
    float scale = sqrtf(2.0f / (float)fan_in);
    for (int i = 0; i < size; ++i)
        weights[i] = (nn_rand_float(rng) * 2.0f - 1.0f) * scale;
}
