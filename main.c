//#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "mlp.h"

#define RANDOM_SCALE 2.0f
#define D_MODEL      8
#define EXTENDING    4
#define LEARNING     50000

#define PRINTING 1 // Prints input and output vectors
#define LOADING  0 // Loads Model from file

#define AFUNC GeLU

#if PRINTING
    #define PRT printf("\n\nInputs:\n");\
    printf("\nParams:\n");\
    for (size_t i = 0; i < model->w_c + model->b_c; i++) {\
        printf("%.3f, ", model->params[i]);\
    }\
    for (size_t i = 0; i < model->layers[0]; i++) {\
        printf("%.3f, ", inputs[i]);\
    }
#else
    #define PRT
#endif

static double ms_between(const struct timespec *start, const struct timespec *end) {
    return (end->tv_sec - start->tv_sec) * 1000.0 +
    (end->tv_nsec - start->tv_nsec) / 1e6;
}

static ptype MSE(const ptype *outputs, const ptype *labels) {
    ptype loss = 0;
    for (size_t i = 0; i < D_MODEL; i++) {
        float val = labels[i] - outputs[i];
        loss += val * val;
    }
    return loss / D_MODEL;
}

void prtout(const Model *model) {
    printf("Outputs:\n");
    for (size_t i = model->b_c-model->layers[model->l_c - 1]; i < model->b_c; i++) {
        printf("%.3f ", model->buffer[i]);
    }
    puts("\n");
}

void prtmdl(const Model *model) {
    const ptype *bias = model->params + model->w_c;
    const ptype *weights = model->params;

    size_t layer = 0;
    for (size_t i = 0; i < model->l_c; i++) {
        size_t prev_layer = layer;
        layer = model->layers[i];

        printf("\n\033[4mLAYER %zu\033[0m\n\033[35mWeights\033[0m:\n", i);

        if (prev_layer) for (size_t j = 0; j < layer; j++) {
            for (size_t k = 0; k < prev_layer; k++) printf("%f ", weights[k]);
            printf("\033[32m%zu\033[0m\n", j);
            weights += prev_layer;
        }

        printf("\033[36mBiases\033[0m:\n");
        for (size_t j = 0; j < layer; j++) {
            printf("%.3f ", bias[j]);
        }
        puts("");
        bias += layer;
    }
}

int main(void) {
    struct timespec t0, t1;

    ptype inputs[D_MODEL] = {1,2,3,4,5,6,7,8};
    ptype labels[D_MODEL] = {3,6,3,8,9,1,0,2};

    clock_gettime(CLOCK_MONOTONIC, &t0);
    Model *model;
    if (LOADING) model = loadModel("model.bin");
    else {
        ntype layers[] = {D_MODEL, D_MODEL * EXTENDING, D_MODEL};
        size_t layer_count = sizeof(layers) / sizeof(layers[0]);

        model = genModel(layers, layer_count, AFUNC);
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

    //PRT
    if (PRINTING) {
        puts("\n\nInputs:");
        for (size_t i = 0; i < D_MODEL; i++) printf("%.3f ", inputs[i]);
    }

    if (PRINTING) {
        puts("\n\nTarget:");
        for (size_t i = 0; i < D_MODEL; i++) printf("%.3f ", labels[i]);
    }

    ptype *outputs = MLP(model, inputs);
    printf("\n\nMSE: %.3f\n", MSE(outputs, labels));

    prtout(model);

    for (size_t i = 0; i < LEARNING; i++) {
        backProp(model, inputs, labels, 0.0001);
        //printf("MSE: %.3f\n", MSE(outputs, labels));
        //prtout(model);
    }
    printf("MSE: %.3f\n", MSE(outputs, labels));
    prtout(model);

    //PRT
    prtmdl(model);

    clock_gettime(CLOCK_MONOTONIC, &t0);
    freeModel(model);
    clock_gettime(CLOCK_MONOTONIC, &t1);

    printf("\n\nMemory release: %.3f ms\n", ms_between(&t0, &t1));

    return 0;
}
