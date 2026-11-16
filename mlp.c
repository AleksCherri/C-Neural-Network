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

// Main perceptron functions
void MLP(Model *model, const ptype *inputs) {
    const ptype *restrict params = model->params;
    ptype *restrict buffer = model->buffer;

    const size_t w_c = model->w_c;
    const ptype *restrict bias = params + w_c;

    ntype layer = model->layers[0];
    ntype prev_layer;

    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < layer; i++) {
        buffer[i] = inputs[i] + bias[i];
    }

    size_t n_ofs = 0, w_ofs = 0;

    for (size_t i = 1; i < model->l_c; i++) {
        prev_layer = layer;
        layer = model->layers[i];

        const ptype *restrict P = params + w_ofs;
        const ptype *restrict B = bias + n_ofs + prev_layer;
        ptype *restrict buf_in  = buffer + n_ofs;
        ptype *restrict buf_out = buf_in + prev_layer;

        #pragma omp parallel for schedule(static)
        for (size_t j = 0; j < layer; j++) {
            const ptype *restrict par = P + j * prev_layer;
            ptype value = B[j];
            for (size_t k = 0; k < prev_layer; k++) {
                value += buf_in[k] * par[k];
            }
            buf_out[j] = model->a_func(value);
        }
        n_ofs += prev_layer;
        w_ofs += prev_layer * layer;
    }
}

void backProp(Model *model, const ptype *labels) {
    const ptype *restrict buffer = model->buffer;
    const ntype D_MODEL = model->layers[0];
    ptype gradient[D_MODEL];

    for (size_t i = 0; i < D_MODEL; i++) {
        ptype val = buffer[model->b_c - model->layers[model->l_c - 1] + i] - labels[i];
        gradient[i] = val;
        //printf("%.3f, ", val);
    }

}

// Utility functions
float *randArray(size_t size, float seed) {
    float *arr = malloc(size * sizeof(float)); if (!arr) return NULL;

    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < size; i++) {
        arr[i] = sin(i / seed);
    }
    return arr;
}

void freeModel(Model *model) {
    if (!model) return;
    free(model->layers);
    free(model->params);
    free(model->buffer);
    free(model);
}

Model *genModel(const ntype *layers, size_t layer_count, ActType a_type) {
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

    model->a_type = a_type;
    model->a_func = getAFunc(a_type);

    model->layers = malloc(layer_count * sizeof(ntype));
    model->params = randArray(b_c + w_c, 3.14f);
    model->buffer = malloc(b_c * sizeof(ptype));

    if (!model->layers || !model->params || !model->buffer) {
        freeModel(model);
        return NULL;
    }

    memcpy(model->layers, layers, layer_count * sizeof(ntype));

    model->l_c = layer_count;
    model->b_c = b_c;
    model->w_c = w_c;

    return model;
}

Model *loadModel(const char *filename) {
    Model *model = malloc(sizeof(Model)); if (!model) return NULL;

    FILE *file = fopen(filename, "rb");
    if (!file) return NULL;

    uint32_t type;

    if (
        fread(&type, sizeof(type), 1, file) != 1
    ) {
        fclose(file);
        return NULL;
    }

    model->a_type = (ActType)type;
    model->a_func = getAFunc(model->a_type);

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

int saveModel(const char *filename, const Model *model) {
    FILE *file = fopen(filename, "wb");
    if (!file) return -1;

    uint32_t type = (uint32_t)model->a_type;

    if (
        fwrite(&type, sizeof(type), 1, file) != 1
    ) {
        fclose(file);
        return -1;
    }
    
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