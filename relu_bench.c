#include <math.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include <immintrin.h>

void __attribute__ ((noinline)) relu_slow(float *restrict X, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        if (X[i] < 0) {
            X[i] = 0;
        }
    }
}

void __attribute__ ((noinline)) relu(float *restrict X, size_t n) {
    __m256 zero = _mm256_set_ps(0., 0., 0., 0., 0., 0., 0., 0.); 
    for (size_t i = 0; i < n; i += 8) {
	__m256 v = _mm256_loadu_ps(X + i);
	v = _mm256_max_ps(v, zero);
	_mm256_storeu_ps(X + i, v);
    }
}

float *random_vec(size_t n) {
    float *vec = malloc(n * sizeof(float));

    for (size_t i = 0; i < n; ++i) {
        float r = (float)rand()/(float)(RAND_MAX);
        vec[i] = (r - 0.5) * 20;
    }

    return vec;
}

float **random_vecs(size_t b, size_t n) {
    float **vecs = malloc(b * sizeof(float *));

    for (size_t i = 0; i < b; ++i) {
        vecs[i] = random_vec(n);
    }

    return vecs;
}

void free_vecs(float **vecs, size_t b) {
    for (size_t i = 0; i < b; ++i) {
        free(vecs[i]);
    }

    free(vecs);
}

const size_t N_ITERATIONS = 100;

// Must be a multiple of 8 to work with ReLU intrinsics.
const size_t ARRAY_SIZE = 3072 * 20 * 32;

int main() {
    // Warmup
    float *v = random_vec(ARRAY_SIZE);
    relu(v, ARRAY_SIZE);
    free(v);

    // Benchmark
    float **vecs = random_vecs(N_ITERATIONS, ARRAY_SIZE);
    float startTime = (float)clock()/CLOCKS_PER_SEC;
    for (size_t i = 0; i < N_ITERATIONS; ++i) {
        relu(vecs[i], ARRAY_SIZE);
    }
    float endTime = (float)clock()/CLOCKS_PER_SEC;
    float time = endTime - startTime;
    free_vecs(vecs, N_ITERATIONS);

    printf("time: %.2f\n", time);
}
