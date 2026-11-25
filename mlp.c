#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "mlp.h"

// Activation functions & related utilities

static inline ptype gelu(ptype x) {
    return x / (1.0f + expf(-1.702f * x));
}

static inline ptype relu(ptype x) {
    return x > 0.0f ? x : 0.0f;
}

static inline ptype relu2(ptype x) {
    return x > 0.0f ? x * x : 0.0f;
}

static ActFunc getAFunc(ActType type) {
    switch (type) {
        case GeLU:  return gelu;
        case ReLU:  return relu;
        case ReLU2: return relu2;
        default:    return NULL;
    }
}

static inline ptype dAFunc(ActType type, ptype x) {
    switch (type) {
        case GeLU: {
            const ptype s = 1.0f / (1.0f + expf(-1.702f * x));
            return s + 1.702f * x * s * (1.0f - s);
        }
        case ReLU:  return x > 0.0f ? 1.0f : 0.0f;
        case ReLU2: return x > 0.0f ? 2.0f * x : 0.0f;
        default:    return 0.0f;
    }
}

// Main perceptron functions

// Multilayered Perceptron Forward
ptype *MLP(Model *model, const ptype *inputs) {
    if (!model) return NULL;

    const ptype *restrict params = model->params;
    ptype *restrict buffer = model->buffer;
    ptype *restrict bufz = model->zbuffr;

    const ptype *restrict bias = params + model->w_c;

    ntype layer = model->layers[0];
    ntype prev_layer;

    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < layer; i++) {
        ptype value = inputs[i] + bias[i];
        bufz[i] = value;
        buffer[i] = value;
    }

    ptype *restrict buf_in = buffer;

    for (size_t i = 1; i < model->l_c; i++) {
        prev_layer = layer;
        layer = model->layers[i];

        bias += prev_layer;
        buffer += prev_layer;
        bufz += prev_layer;

        #pragma omp parallel for schedule(static)
        for (size_t j = 0; j < layer; j++) {
            const ptype *restrict par = params + j * prev_layer;
            ptype value = bias[j];
            for (size_t k = 0; k < prev_layer; k++) {
                value += buf_in[k] * par[k];
            }
            bufz[j] = value;
            buffer[j] = model->a_func(value);
        }
        buf_in += prev_layer;
        params += prev_layer * layer;
    }

    return buffer;
}

// Backprogapation function
void backProp(Model *model, const ptype *inputs, const ptype *labels, float learning_rate) {
    if (!model) return;

    ptype *restrict params = model->params + model->w_c;
    ptype *restrict buffer = model->buffer;
    ptype *restrict bias = model->params + model->w_c;

    const int D_MODEL = (int)model->layers[model->l_c - 1];
    const size_t ofs = model->b_c - model->layers[model->l_c - 1];

    ptype deltas[model->b_c];
    ptype *delout = (ptype *)deltas + ofs;
    ptype *output = MLP(model, inputs); //model->buffer + ofs;
    ptype *bufin = model->zbuffr + ofs;


    #pragma omp parallel for schedule(static)
    for (int i = D_MODEL - 1; i >= 0; i--) {
        delout[i] = (labels[i] - output[i]) * dAFunc(model->a_type, bufin[i]);
    }
    //for (int i = D_MODEL - 1; i >= 0; i--) printf("%.3f, ", delout[i]);

    int next_layer, layer = D_MODEL;

    for (int i = (int)model->l_c - 2; i > 0; i--) {
        next_layer = layer;
        layer = (int)model->layers[i];

        params -= next_layer * layer;
        ptype *delin = delout;
        delout -= layer;
        bufin -= layer;

        #pragma omp parallel for schedule(static)
        for (int j = layer - 1; j >= 0; j--) {
            ptype val = 0;
            const ptype *restrict par = params + j;
            for (int k = next_layer - 1; k >= 0; k--)
                val += delin[k] * par[k * layer];
            delout[j] = val * dAFunc(model->a_type, bufin[j]);
        }
    }

    params = model->params;
    bias   = model->params + model->w_c + model->layers[0];
    buffer = model->buffer;
    delout = deltas + model->layers[0];

    size_t prev_layer = model->layers[0];
    for (size_t i = 1; i < model->l_c; i++) {
        const size_t cur_layer = model->layers[i];
        #pragma omp parallel for schedule(static)
        for (size_t j = 0; j < cur_layer; j++) {
            const ptype adj = delout[j] * learning_rate;
            bias[j] += adj;
            ptype *restrict par = params + j * prev_layer;
            for (size_t k = 0; k < prev_layer; k++)
                par[k] += adj * buffer[k];
        }
        params += prev_layer * cur_layer;
        bias   += cur_layer;
        buffer += prev_layer;
        delout += cur_layer;
        prev_layer = cur_layer;
    }
}

// Utility functions

float *randArray(size_t size, float seed) {
    if (size == 0) return NULL;

    float *arr = malloc(size * sizeof(*arr));
    if (!arr) return NULL;

    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < size; i++) {
        uint32_t state = ((uint32_t)(seed * 1000.0f) ^ (uint32_t)(i * 2654435761u)) | 1u;
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        arr[i] = 2.0f * (ptype)state / (ptype)UINT32_MAX - 1.0f;
    }

    return arr;
}

void freeModel(Model *model) {
    if (!model) return;
    if (model->layers) free(model->layers);
    if (model->params) free(model->params);
    if (model->buffer) free(model->buffer);
    if (model->zbuffr) free(model->zbuffr);
    free(model);
}

Model *allocModel(size_t layer_count, size_t b_c, size_t w_c) {
    Model *model = malloc(sizeof(Model)); if (!model) goto fail;

    model->layers = malloc(layer_count * sizeof(ntype));
    model->params = randArray(b_c + w_c, 3.14f);
    model->buffer = malloc(b_c * sizeof(ptype));
    model->zbuffr = malloc(b_c * sizeof(ptype));

    if (
        !model->layers || !model->params ||
        !model->buffer || !model->zbuffr
    ) goto fail;

    model->l_c = layer_count;
    model->b_c = b_c;
    model->w_c = w_c;

    return model;

    fail:
    freeModel(model);
    printf("\nMEMORY ALLOCATION FAILED!\n");
    return NULL;
}

Model *genModel(const ntype *layers, size_t layer_count, ActType a_type) {
    if (layer_count < 2) return NULL;
    size_t b_c = layers[layer_count - 1], w_c = 0;
    size_t layer, next_layer;
    for (size_t i = 0; i < layer_count - 1; i++) {
        layer = layers[i];
        next_layer = layers[i + 1];
        if (layer == 0 || next_layer == 0) return NULL;
        b_c += layer;
        w_c += (size_t)layer * next_layer;
    }

    Model *model = allocModel(layer_count, b_c, w_c); if (!model) goto fail;

    model->a_type = a_type;
    model->a_func = getAFunc(a_type);
    if (!model->a_func) goto fail;

    memcpy(model->layers, layers, layer_count * sizeof(ntype));

    return model;

    fail:
    printf("\nFAILED TO GEN!\n");
    return NULL;
}

Model *loadModel(const char *filename) {

    FILE *file = fopen(filename, "rb");
    if (!file) {puts("ERROR: No file"); goto fail;}

    size_t l_c;
    size_t b_c;
    size_t w_c;
    uint32_t type;

    size_t size = sizeof(size_t);
    if (
        fread(&l_c, size, 1, file) != 1 ||
        fread(&w_c, size, 1, file) != 1 ||
        fread(&b_c, size, 1, file) != 1 ||
        fread(&type, sizeof(type), 1, file) != 1
    ) {puts("ERROR: No hyperparameters"); goto fail;}

    Model *model = allocModel(l_c, b_c, w_c); if (!model) goto fail;

    if (type > ReLU2) {puts("ERROR: No afunc type"); goto fail;}
    model->a_type = (ActType)type;
    model->a_func = getAFunc(model->a_type);

    if (
        model->l_c <= 0 || model->b_c <= 0 ||
        model->w_c <= 0 || model->l_c > 1024
    ) {printf("ERROR: Counters are out of bounds: %zu, %zu\n", model->l_c, model->w_c); goto fail;}

    if (
        fread(model->layers, sizeof(ntype), model->l_c, file)
        != (size_t)model->l_c
    ) {puts("ERROR: Layer count mismatch"); goto fail;}

    size = model->b_c + model->w_c;
    if (
        fread(model->params, sizeof(ptype), size, file)
        != size
    ) {puts("ERROR: Parameter count mismatch"); goto fail;}

    fclose(file);
    return model;

    fail:
    if (file) fclose(file);
    printf("\nFAILED TO LOAD!\n");
    return NULL;
}

int saveModel(const char *filename, const Model *model) {
    FILE *file = fopen(filename, "wb");
    if (!file) {puts("ERROR: No file"); goto fail;}

    uint32_t type = (uint32_t)model->a_type;
    
    size_t size = sizeof(size_t);
    if (
        fwrite(&model->l_c, size, 1, file) != 1 ||
        fwrite(&model->w_c, size, 1, file) != 1 ||
        fwrite(&model->b_c, size, 1, file) != 1 ||
        fwrite(&type, sizeof(type), 1, file) != 1
    ) {puts("ERROR: Cannot write hyperparameters"); goto fail;}

    if (model->l_c > 0) {
        if (fwrite(model->layers, sizeof(ntype), model->l_c, file)
        != (size_t)model->l_c) {puts("ERROR: Layer count mismatch"); goto fail;}
    }

    if (model->b_c > 0 && model->w_c > 0) {
        size_t p_c = model->b_c + model->w_c;
        if (fwrite(model->params, sizeof(ptype), p_c, file) != p_c)
        {puts("ERROR: Parameter count mismatch"); goto fail;}
    }

    fclose(file);
    return 0;

    fail:
    if (file) fclose(file);
    printf("\nFAILED TO SAVE!\n");
    return -1;
}
