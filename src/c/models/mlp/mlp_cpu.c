#include "mlp.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <stdint.h>
#include <time.h>

#ifdef _OPENMP
#include <omp.h>
#include <unistd.h>

/* Detect physical core count for optimal OMP threads.
 * Hyperthreaded cores share FPU, so for compute-bound GEMM
 * using only physical cores avoids contention.
 * Falls back to omp_get_max_threads() if detection fails. */
static int detect_physical_cores(void) {
    /* Try Linux sysfs: count unique physical core IDs */
    int max_cpus = (int)sysconf(_SC_NPROCESSORS_ONLN);
    if (max_cpus <= 0) max_cpus = 1;

    /* Read core_id for each CPU to count unique physical cores */
    int seen[256] = {0};  /* bitmap: seen[core_id] */
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
    /* Only auto-set if user hasn't explicitly set OMP_NUM_THREADS */
    if (!getenv("OMP_NUM_THREADS")) {
        int cores = detect_physical_cores();
        omp_set_num_threads(cores);
    }
}
#endif

#ifdef USE_BLAS
#include <cblas.h>
#endif

/* ------------------------------------------------------------------ */
/*  GEMM: cblas when available, cache-tiled OpenMP fallback           */
/* ------------------------------------------------------------------ */

/* Tile size tuned for L1 cache (~32KB): 3 tiles of 64x64 floats = 48KB. */
#define TILE 64

/* Minimum total FLOPs (M*N*K) to justify OpenMP thread overhead.
 * With persistent thread team (libgomp), overhead is ~1-5μs per region.
 * At ~4 GFLOP/s single-core, 100K FLOPs/thread = ~25μs — amortizes easily. */
#define OMP_FLOP_THRESHOLD 100000L

/* Minimum elements for elementwise OpenMP (bias+relu, softmax, SGD).
 * Each element ~2-5 FLOPs; 50K elements/thread = ~25-60μs of work. */
#define OMP_ELEM_THRESHOLD 200000L

#ifndef USE_BLAS

/* C[M,N] = A[M,K] @ B[K,N]  — cache-friendly i-k-j with optional tiling */
static void sgemm_nn(int M, int N, int K,
                     const float * restrict A,
                     const float * restrict B,
                     float * restrict C) {
    long flops = (long)M * N * K;

    if (K <= TILE && N <= TILE) {
        /* Small matrices: simple loop, let compiler auto-vectorize the j-loop.
         * With -O3 -march=native, gcc generates AVX2 FMA for the inner loop. */
        #pragma omp parallel for schedule(static) if(flops > OMP_FLOP_THRESHOLD)
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
        /* Large matrices: tiled for L1 cache residency */
        memset(C, 0, (size_t)M * N * sizeof(float));
        #pragma omp parallel for schedule(static) if(flops > OMP_FLOP_THRESHOLD)
        for (int i0 = 0; i0 < M; i0 += TILE) {
            int imax = (i0 + TILE < M) ? i0 + TILE : M;
            for (int k0 = 0; k0 < K; k0 += TILE) {
                int kmax = (k0 + TILE < K) ? k0 + TILE : K;
                for (int j0 = 0; j0 < N; j0 += TILE) {
                    int jmax = (j0 + TILE < N) ? j0 + TILE : N;
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

/* C[M,N] = A^T[M,K] @ B[K,N],  A stored as [K,M] */
static void sgemm_tn(int M, int N, int K,
                     const float * restrict A,
                     const float * restrict B,
                     float * restrict C) {
    long flops = (long)M * N * K;
    #pragma omp parallel for schedule(static) if(flops > OMP_FLOP_THRESHOLD)
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

/* C[M,N] = A[M,K] @ B^T[K,N],  B stored as [N,K] */
static void sgemm_nt(int M, int N, int K,
                     const float * restrict A,
                     const float * restrict B,
                     float * restrict C) {
    long flops = (long)M * N * K;
    #pragma omp parallel for schedule(static) if(flops > OMP_FLOP_THRESHOLD)
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

static void sgemm_nn(int M, int N, int K,
                     const float *A, const float *B, float *C) {
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                M, N, K, 1.0f, A, K, B, N, 0.0f, C, N);
}

static void sgemm_tn(int M, int N, int K,
                     const float *A, const float *B, float *C) {
    cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                M, N, K, 1.0f, A, M, B, N, 0.0f, C, N);
}

static void sgemm_nt(int M, int N, int K,
                     const float *A, const float *B, float *C) {
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                M, N, K, 1.0f, A, K, B, K, 0.0f, C, N);
}

#endif /* USE_BLAS */

/* ------------------------------------------------------------------ */
/*  PRNG for weight init                                              */
/* ------------------------------------------------------------------ */

static uint32_t xorshift32(uint32_t *state) {
    uint32_t x = *state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *state = x;
    return x;
}

static float rand_float(uint32_t *state) {
    return (float)(xorshift32(state) & 0x7FFFFFFF) / (float)0x7FFFFFFF;
}

/* ------------------------------------------------------------------ */
/*  Init / Free                                                       */
/* ------------------------------------------------------------------ */

void mlp_initialize(MLP *mlp, int input_size, int hidden_size, int output_size) {
    mlp->input_size = input_size;
    mlp->hidden_size = hidden_size;
    mlp->output_size = output_size;

    mlp->input_weights  = (float *)malloc(input_size  * hidden_size * sizeof(float));
    mlp->output_weights = (float *)malloc(hidden_size * output_size * sizeof(float));
    mlp->hidden_biases  = (float *)calloc(hidden_size, sizeof(float));
    mlp->output_biases  = (float *)calloc(output_size, sizeof(float));

    if (!mlp->input_weights || !mlp->output_weights ||
        !mlp->hidden_biases || !mlp->output_biases) {
        fprintf(stderr, "Failed to allocate MLP parameters.\n");
        exit(EXIT_FAILURE);
    }

    uint32_t rng = (uint32_t)time(NULL);
    float s1 = sqrtf(2.0f / (float)input_size);
    for (int i = 0; i < input_size * hidden_size; ++i)
        mlp->input_weights[i] = (rand_float(&rng) * 2.0f - 1.0f) * s1;

    float s2 = sqrtf(2.0f / (float)hidden_size);
    for (int i = 0; i < hidden_size * output_size; ++i)
        mlp->output_weights[i] = (rand_float(&rng) * 2.0f - 1.0f) * s2;
}

void mlp_free(MLP *mlp) {
    free(mlp->input_weights);
    free(mlp->output_weights);
    free(mlp->hidden_biases);
    free(mlp->output_biases);
}

/* ------------------------------------------------------------------ */
/*  Batched forward pass                                              */
/* ------------------------------------------------------------------ */

static void forward_batch(const MLP *mlp, const float *X, int bs,
                          float *hidden, float *output) {
    int in  = mlp->input_size;
    int hid = mlp->hidden_size;
    int out = mlp->output_size;

    /* hidden = X @ W1 */
    sgemm_nn(bs, hid, in, X, mlp->input_weights, hidden);

    /* Add bias + ReLU */
    #pragma omp parallel for schedule(static) if(bs * hid > OMP_ELEM_THRESHOLD)
    for (int i = 0; i < bs; ++i) {
        float *h = hidden + i * hid;
        for (int j = 0; j < hid; ++j) {
            h[j] += mlp->hidden_biases[j];
            if (h[j] < 0.0f) h[j] = 0.0f;
        }
    }

    /* output = hidden @ W2 */
    sgemm_nn(bs, out, hid, hidden, mlp->output_weights, output);

    /* Add bias + softmax (per sample) */
    #pragma omp parallel for schedule(static) if((long)bs * out > OMP_ELEM_THRESHOLD)
    for (int i = 0; i < bs; ++i) {
        float *o = output + i * out;

        float max_val = -1e30f;
        for (int j = 0; j < out; ++j) {
            o[j] += mlp->output_biases[j];
            if (o[j] > max_val) max_val = o[j];
        }

        float sum = 0.0f;
        for (int j = 0; j < out; ++j) {
            o[j] = expf(o[j] - max_val);
            sum += o[j];
        }
        for (int j = 0; j < out; ++j)
            o[j] /= sum;
    }
}

/* ------------------------------------------------------------------ */
/*  Training: batched GEMM forward + backward                        */
/* ------------------------------------------------------------------ */

void mlp_train(MLP *mlp, float *inputs, int *targets, int num_samples,
               int batch_size, int num_epochs, float learning_rate) {
    int in  = mlp->input_size;
    int hid = mlp->hidden_size;
    int out = mlp->output_size;

    float *hidden   = (float *)malloc(batch_size * hid * sizeof(float));
    float *output   = (float *)malloc(batch_size * out * sizeof(float));
    float *d_output = (float *)malloc(batch_size * out * sizeof(float));
    float *d_hidden = (float *)malloc(batch_size * hid * sizeof(float));
    float *grad_W1  = (float *)malloc(in  * hid * sizeof(float));
    float *grad_W2  = (float *)malloc(hid * out * sizeof(float));
    float *grad_b1  = (float *)malloc(hid * sizeof(float));
    float *grad_b2  = (float *)malloc(out * sizeof(float));

    if (!hidden || !output || !d_output || !d_hidden ||
        !grad_W1 || !grad_W2 || !grad_b1 || !grad_b2) {
        fprintf(stderr, "Failed to allocate training workspace.\n");
        exit(EXIT_FAILURE);
    }

    int num_batches = (num_samples + batch_size - 1) / batch_size;

    for (int epoch = 0; epoch < num_epochs; ++epoch) {
        float epoch_loss = 0.0f;

        for (int batch = 0; batch < num_batches; ++batch) {
            int start = batch * batch_size;
            int bs = batch_size;
            if (start + bs > num_samples) bs = num_samples - start;

            const float *X = inputs + start * in;
            const int *Y   = targets + start;

            /* ---- Forward ---- */
            forward_batch(mlp, X, bs, hidden, output);

            /* ---- Loss + d_output = softmax - one_hot ---- */
            float batch_loss = 0.0f;
            #pragma omp parallel for reduction(+:batch_loss) schedule(static) if((long)bs * out > OMP_ELEM_THRESHOLD)
            for (int i = 0; i < bs; ++i) {
                float *o = output   + i * out;
                float *d = d_output + i * out;
                batch_loss -= logf(o[Y[i]] + 1e-7f);
                for (int j = 0; j < out; ++j)
                    d[j] = o[j] - ((Y[i] == j) ? 1.0f : 0.0f);
            }
            epoch_loss += batch_loss;

            /* ---- Backward ---- */

            /* grad_W2[hid,out] = hidden^T @ d_output */
            sgemm_tn(hid, out, bs, hidden, d_output, grad_W2);

            /* grad_b2 = column sum of d_output */
            memset(grad_b2, 0, out * sizeof(float));
            for (int i = 0; i < bs; ++i) {
                const float *d = d_output + i * out;
                for (int j = 0; j < out; ++j)
                    grad_b2[j] += d[j];
            }

            /* d_hidden = d_output @ W2^T, then ReLU mask */
            sgemm_nt(bs, hid, out, d_output, mlp->output_weights, d_hidden);

            #pragma omp parallel for schedule(static) if((long)bs * hid > OMP_ELEM_THRESHOLD)
            for (int i = 0; i < bs * hid; ++i)
                if (hidden[i] <= 0.0f) d_hidden[i] = 0.0f;

            /* grad_W1[in,hid] = X^T @ d_hidden */
            sgemm_tn(in, hid, bs, X, d_hidden, grad_W1);

            /* grad_b1 = column sum of d_hidden */
            memset(grad_b1, 0, hid * sizeof(float));
            for (int i = 0; i < bs; ++i) {
                const float *d = d_hidden + i * hid;
                for (int j = 0; j < hid; ++j)
                    grad_b1[j] += d[j];
            }

            /* ---- SGD update ---- */
            float lr_s = learning_rate / (float)bs;

            #pragma omp parallel for schedule(static) if((long)in * hid > OMP_ELEM_THRESHOLD)
            for (int i = 0; i < in * hid; ++i)
                mlp->input_weights[i] -= lr_s * grad_W1[i];

            #pragma omp parallel for schedule(static) if((long)hid * out > OMP_ELEM_THRESHOLD)
            for (int i = 0; i < hid * out; ++i)
                mlp->output_weights[i] -= lr_s * grad_W2[i];

            for (int i = 0; i < hid; ++i)
                mlp->hidden_biases[i] -= lr_s * grad_b1[i];
            for (int i = 0; i < out; ++i)
                mlp->output_biases[i] -= lr_s * grad_b2[i];
        }

        if (epoch % 100 == 0)
            printf("  Epoch %4d  loss: %.4f\n", epoch, epoch_loss / num_samples);
    }

    free(hidden);  free(output);
    free(d_output); free(d_hidden);
    free(grad_W1); free(grad_W2);
    free(grad_b1); free(grad_b2);
}

/* ------------------------------------------------------------------ */
/*  Evaluation: batched forward                                       */
/* ------------------------------------------------------------------ */

void mlp_evaluate(MLP *mlp, float *inputs, int *targets, int num_samples,
                  float *loss, float *accuracy) {
    int hid = mlp->hidden_size;
    int out = mlp->output_size;

    float *hidden = (float *)malloc(num_samples * hid * sizeof(float));
    float *output = (float *)malloc(num_samples * out * sizeof(float));

    if (!hidden || !output) {
        fprintf(stderr, "Failed to allocate evaluation workspace.\n");
        exit(EXIT_FAILURE);
    }

    forward_batch(mlp, inputs, num_samples, hidden, output);

    float total_loss = 0.0f;
    int correct = 0;

    #pragma omp parallel for reduction(+:total_loss,correct) schedule(static) if((long)num_samples * out > OMP_ELEM_THRESHOLD)
    for (int i = 0; i < num_samples; ++i) {
        const float *o = output + i * out;
        total_loss -= logf(o[targets[i]] + 1e-7f);

        int pred = 0;
        float mx = o[0];
        for (int j = 1; j < out; ++j) {
            if (o[j] > mx) { mx = o[j]; pred = j; }
        }
        if (pred == targets[i]) correct++;
    }

    *loss = total_loss / num_samples;
    *accuracy = (float)correct / num_samples * 100.0f;

    free(hidden);
    free(output);
}
