//#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "mlp.h"

#define RANDOM_SCALE 2.0f
#define D_MODEL      4
#define EXTENDING    4

#define PRINTING 0 // Prints input and output vectors
#define LOADING  1 // Loads Model from file

#if PRINTING
    #define PRT printf("\n\nInputs:\n");\
    printf("\nParams:\n");\
    for (size_t i = 0; i < model->w_c; i++) {\
        printf("%.3f, ", model->params[i]);\
    }\
    for (size_t i = 0; i < model->layers[0]; i++) {\
        printf("%.3f, ", inputs[i]);\
    }\
    printf("\n\nOutputs:\n");\
    for (size_t i = model->b_c-model->layers[model->l_c - 1]; i < model->b_c; i++) {\
        printf("%.3f, ", model->buffer[i]);\
    }
#else
    #define PRT
#endif

static double ms_between(const struct timespec *start, const struct timespec *end) {
    return (end->tv_sec - start->tv_sec) * 1000.0 +
    (end->tv_nsec - start->tv_nsec) / 1e6;
}

int main(void) {
    struct timespec t0, t1;

    ptype inputs[D_MODEL] = {1,2,3,4};
    ptype labels[D_MODEL] = {4,3,2,1};
    ptype outputs[D_MODEL];

    clock_gettime(CLOCK_MONOTONIC, &t0);
    Model *model;
    if (LOADING) model = loadModel("model.bin");
    else {
        ntype layers[] = {D_MODEL, D_MODEL * EXTENDING, D_MODEL};
        size_t layer_count = sizeof(layers) / sizeof(layers[0]);

        model = genModel(layers, layer_count, ReLU);
    }
    clock_gettime(CLOCK_MONOTONIC, &t1);

    if (!model) {printf("\nNO MODEL!\n"); return 1;}

    printf("\nModel %s: %.3f ms", LOADING ? "load" : "gen", ms_between(&t0, &t1));

    if (!LOADING) {
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
    if (PRINTING) {
        printf("\n\nTarget:\n");
        for (size_t i = 0; i < D_MODEL; i++) printf("%.3f, ", labels[i]);
    }

    for (size_t i = 0; i < D_MODEL; i++) {
        outputs[i] = model->buffer[model->b_c - model->layers[model->l_c - 1] + i];
    }

    ptype loss = 0;
    for (size_t i = 0; i < D_MODEL; i++) {
        float val = labels[i] - outputs[i];
        loss += val * val;
    }
    loss /= D_MODEL;

    printf("\n\nMSE: %.3f\n", loss);

    backProp(model, inputs, labels, 0.1);

    clock_gettime(CLOCK_MONOTONIC, &t0);
    freeModel(model);
    clock_gettime(CLOCK_MONOTONIC, &t1);

    printf("\n\nMemory release: %.3f ms\n", ms_between(&t0, &t1));

    return 0;
}
