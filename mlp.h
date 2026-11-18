#include <stddef.h>
#include <stdint.h>

#ifndef MLP_H
#define MLP_H

typedef uint32_t ntype;
typedef float    ptype;

typedef enum {
    GeLU  = 0,
    ReLU  = 1,
    ReLU2 = 2
} ActType;

typedef ptype (*ActFunc)(ptype x);

typedef struct {
    ActType a_type; /* Index of activation function */
    ActFunc a_func; /* Activation function */
    size_t l_c;     /* Number of layers */
    ntype *layers;  /* Layer structure */
    size_t b_c;     /* Number of biases (nodes) */
    size_t w_c;     /* Number of weights */
    ptype *params;  /* Parameters weights[w_c] + biases[b_c] */
    ptype *buffer;  /* Buffer of activations for backpropagation */
} Model;

float *randArray(size_t size, float seed);
ptype *MLP(Model *model, const ptype *inputs);
void   backProp(Model *model, const ptype *inputs, const ptype *labels, float learning_rate);
void   freeModel(Model *model);
Model *genModel(const ntype *layers, size_t layer_count, ActType a_type);
Model *loadModel(const char *filename);
int    saveModel(const char *filename, const Model *model);

#endif
