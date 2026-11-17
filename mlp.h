#include <stddef.h>
#include <stdint.h>

#ifndef MLP_H
#define MLP_H

#define ntype unsigned int
#define ptype float

typedef enum {
    GeLU  = 0,
    ReLU  = 1,
    ReLU2 = 2
} ActType;

typedef ptype (*ActFunc)(ptype x);

typedef struct {
    ActType a_type;
    ActFunc a_func;
    size_t l_c;
    ntype *layers;
    size_t b_c;
    size_t w_c;
    ptype *params; // weights[w_c] + biases[b_c]
    ptype *buffer;
} Model;

float *randArray(size_t size, float seed);
void MLP(Model *model, const ptype *inputs);
void backProp(Model *model, const ptype *labels);
void freeModel(Model *model);
Model *genModel(const ntype *layers, size_t layer_count, ActType a_type);
Model *loadModel(const char *filename);
int saveModel(const char *filename, const Model *model);

#endif
