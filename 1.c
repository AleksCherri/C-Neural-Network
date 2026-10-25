#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

#define RANDOM_SCALE 2.0f
#define D_MODEL      1024
#define EXTENDING    1024

#define ntype unsigned int
#define ptype float

#define PRINTING 0 // Prints input and output vectors
#define VFAURG   0 // Very Fast And Unsafe Random Generator™
#define SAVING   0 // Saves Model to file
#define LOADING  1 // Loads Model from file

#if PRINTING
    #define PRT printf("\n\nInputs:\n");for(size_t i=0;i<model->layers[0];i++){printf("%.3f, ",inputs[i]);}printf("\n\nOutputs:\n");for(size_t i=model->b_c-model->layers[model->l_c-1];i<model->b_c;i++){printf("%.3f, ",model->buffer[i]);}
#else
    #define PRT
#endif

#if VFAURG && defined(_WIN32)
    #define PAR _Pragma("omp parallel for schedule(static)")
#else
    #define PAR
#endif

static double ms_between(const struct timespec *start, const struct timespec *end) {
    return (end->tv_sec - start->tv_sec) * 1000.0 +
    (end->tv_nsec - start->tv_nsec) / 1e6;
}

typedef struct {
    size_t l_c;
    ntype *layers;
    size_t b_c;
    size_t w_c;
    ptype *params; // weights[w_c] + biases[b_c]
    ptype *buffer;
} Model;

void freeModel(Model *model) {
    if (!model) return;
    free(model->layers);
    free(model->params);
    free(model->buffer);
    free(model);
}

// float *rnum(size_t size, float seed) {
//     float arr[size];
//     for (size_t i = 0; i < size; i++) {
//         arr[i];
//     }
//     return arr;
// }

static inline ptype gelu(ptype x) {
    return x / (1.0f + expf(-1.702f * x));
}

void MLP(Model *model, ptype *inputs) {
    const ptype *restrict params = model->params;
    ptype *restrict buffer = model->buffer;

    const size_t w_c = model->w_c;
    const ptype *restrict bias = params + w_c;

    ntype layer = model->layers[0];
    ntype prev_layer;

    #pragma omp parallel for
    for (size_t i = 0; i < layer; i++) {
        buffer[i] = inputs[i] + bias[i];
    }

    size_t n_ofs = 0, w_ofs = 0;

    for (size_t i = 1; i < model->l_c; i++) {
        prev_layer = layer;
        layer = model->layers[i];
        size_t np = n_ofs + prev_layer;
        #pragma omp parallel for
        for (size_t j = 0; j < layer; j++) {
            size_t idx = np + j;
            size_t wp = w_ofs + j * prev_layer;
            ptype value = params[w_c + idx];
            for (size_t k = 0; k < prev_layer; k++) {
                value += buffer[n_ofs + k] * params[wp + k];
            }
            buffer[idx] = gelu(value);
        }
        n_ofs += prev_layer;
        w_ofs += prev_layer * layer;
    }
}

Model *genModel(const ntype *layers, size_t layer_count) {
    if (layer_count == 0) {return NULL;}
    size_t b_c = layers[layer_count - 1], w_c = 0;
    size_t layer, next_layer;
    for (size_t i = 0; i < layer_count - 1; i++) {
        layer = layers[i];
        next_layer = layers[i + 1];
        if (layer == 0 || next_layer == 0) {return NULL;}
        b_c += layer;
        w_c += layer * next_layer;
    }

    Model *model = malloc(sizeof(Model)); if (!model) return NULL;

    model->layers = malloc(layer_count * sizeof(ntype));
    model->params = malloc((w_c + b_c) * sizeof(ptype));
    model->buffer = malloc(b_c * sizeof(ptype));

    if (!model->layers || !model->params || !model->buffer) {
        freeModel(model);
        return NULL;
    }

    memcpy(model->layers, layers, layer_count * sizeof(ntype));

    model->l_c = layer_count;
    model->b_c = b_c;
    model->w_c = w_c;

    PAR
    for (size_t i = 0; i < b_c + w_c; i++) {
        model->params[i] = ((ptype)rand() / RAND_MAX - 0.5) * RANDOM_SCALE;
    }

    return model;
}

int saveModel(const char *filename, const Model *model) {
    FILE *file = fopen(filename, "wb");
    if (!file) return -1;

    size_t size = sizeof(size_t);
    if (
        fwrite(&model->l_c, size, 1, file) != 1 ||
        fwrite(&model->b_c, size, 1, file) != 1 ||
        fwrite(&model->w_c, size, 1, file) != 1
    ) {
        fclose(file);
        return -1;
    }

    if (model->l_c > 0) {
        if (fwrite(model->layers, sizeof(ntype), model->l_c, file)
        != (size_t)model->l_c) {
            fclose(file);
            return -1;
        }
    }

    if (model->b_c > 0 && model->w_c > 0) {
        size_t p_c = model->b_c + model->w_c;
        if (fwrite(model->params, sizeof(ptype), p_c, file) != p_c) {
            fclose(file);
            return -1;
        }
    }

    fclose(file);
    return 0;
}

Model *loadModel(const char *filename) {

    Model *model = malloc(sizeof(Model)); if (!model) return NULL;

    FILE *file = fopen(filename, "rb");
    if (!file) return NULL;

    size_t size = sizeof(size_t);
    if (
        fread(&model->l_c, size, 1, file) != 1 ||
        fread(&model->b_c, size, 1, file) != 1 ||
        fread(&model->w_c, size, 1, file) != 1
    ) {
        fclose(file);
        return NULL;
    }

    if (model->l_c <= 0 || model->b_c <= 0 || model->w_c <= 0) {
        fclose(file);
        return NULL;
    }

    model->layers = malloc(model->l_c * sizeof(ntype));
    model->params = malloc((model->w_c + model->b_c) * sizeof(ptype));
    model->buffer = malloc(model->b_c * sizeof(ptype));

    if (!model->layers || !model->params || !model->buffer) {
        freeModel(model);
        fclose(file);
        return NULL;
    }

    if (
        fread(model->layers, sizeof(ntype), model->l_c, file)
        != (size_t)model->l_c
    ) {
        freeModel(model);
        fclose(file);
        return NULL;
    }

    size = model->b_c + model->w_c;
    if (
        fread(model->params, sizeof(ptype), size, file)
        != size
    ) {
        freeModel(model);
        fclose(file);
        return NULL;
    }

    fclose(file);
    return model;
}

// gcc -std=c17 -Wall -Wextra -O3 -march=native -flto -fopenmp -ffast-math 1.c -o a -lm && "./a"
int main(void) {
    struct timespec t0, t1;

    ptype inputs[D_MODEL];

    for (size_t i = 0; i < D_MODEL; i++) {
        inputs[i] = (ptype)i;
    }

    clock_gettime(CLOCK_MONOTONIC, &t0);
    Model *model;
    if (LOADING) model = loadModel("model.bin");
    else {
        ntype layers[] = {D_MODEL, D_MODEL * EXTENDING, D_MODEL};
        size_t layer_count = sizeof(layers) / sizeof(layers[0]);

        model = genModel(layers, layer_count);
    }
    clock_gettime(CLOCK_MONOTONIC, &t1);

    if (!model) return 1;

    printf("\nModel gen/load: %.3f ms", ms_between(&t0, &t1));

    if (SAVING) {
        clock_gettime(CLOCK_MONOTONIC, &t0);
        if (saveModel("model.bin", model) != 0) return EXIT_FAILURE;
        clock_gettime(CLOCK_MONOTONIC, &t1);
        printf("\nModel saved: %.3f ms", ms_between(&t0, &t1));
    }

    printf("\nLayer count:  %zu", model->l_c);
    printf("\nWeight count: %zu", model->w_c);
    printf("\nBias count:   %zu", model->b_c);

    clock_gettime(CLOCK_MONOTONIC, &t0);
    MLP(model, inputs);
    clock_gettime(CLOCK_MONOTONIC, &t1);

    printf("\n\nForward pass: %.3f ms", ms_between(&t0, &t1));

    PRT
    //printf("\n\nPress Enter: "); getchar();

    clock_gettime(CLOCK_MONOTONIC, &t0);
    freeModel(model);
    clock_gettime(CLOCK_MONOTONIC, &t1);

    printf("\n\nMemory release: %.3f ms\n", ms_between(&t0, &t1));

    return 0;
}
