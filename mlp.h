#ifndef MLP_H
#define MLP_H

#define ntype unsigned int
#define ptype float

typedef struct {
    size_t l_c;
    ntype *layers;
    size_t b_c;
    size_t w_c;
    ptype *params; // weights[w_c] + biases[b_c]
    ptype *buffer;
} Model;

float *randArray(size_t size, float seed);
ptype gelu(ptype x);
ptype relu(ptype x);
ptype relu2(ptype x);
void MLP(Model *model, const ptype *inputs);
void backProp(Model *model, const ptype *labels);
void freeModel(Model *model);
Model *genModel(const ntype *layers, size_t layer_count);
Model *loadModel(const char *filename);
int saveModel(const char *filename, const Model *model);

#endif