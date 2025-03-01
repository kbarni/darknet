#include "l2norm_layer.h"
#include "activations.h"
#include "blas.h"
#include "dark_cuda.h"

#include <float.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>

layer make_l2norm_layer(int batch, int inputs)
{
    fprintf(stderr, "l2norm                                         %4d\n",  inputs);
    layer l = {(LAYER_TYPE)0};
    l.type = L2NORM;
    l.batch = batch;
    l.inputs = inputs;
    l.outputs = inputs;
    l.output = (float*)calloc(inputs*batch, sizeof(float));
    l.scales = (float*)calloc(inputs*batch, sizeof(float));
    l.delta = (float*)calloc(inputs*batch, sizeof(float));

    l.forward = forward_l2norm_layer;
    l.backward = backward_l2norm_layer;
    #ifdef GPU
    l.forward_gpu = forward_l2norm_layer_gpu;
    l.backward_gpu = backward_l2norm_layer_gpu;

    l.output_gpu = cuda_make_array(l.output, inputs*batch); 
    l.scales_gpu = cuda_make_array(l.output, inputs*batch); 
    l.delta_gpu = cuda_make_array(l.delta, inputs*batch); 
    #endif
    return l;
}

void forward_l2norm_layer(const layer l, network_state state)
{
    copy_cpu(l.outputs*l.batch, state.input, 1, l.output, 1);
    l2normalize_cpu(l.output, l.scales, l.batch, l.out_c, l.out_w*l.out_h);
}

void backward_l2norm_layer(const layer l, network_state state)
{
    axpy_cpu(l.inputs*l.batch, 1, l.delta, 1, state.delta, 1);
}

#ifdef GPU

void forward_l2norm_layer_gpu(const layer l, network_state state)
{
    copy_ongpu(l.outputs*l.batch, state.input, 1, l.output_gpu, 1);
    l2normalize_gpu(l.output_gpu, l.scales_gpu, l.batch, l.out_c, l.out_w*l.out_h);
}

void backward_l2norm_layer_gpu(const layer l, network_state state)
{
    axpy_ongpu(l.batch*l.inputs, 1, l.scales_gpu, 1, l.delta_gpu, 1);
    axpy_ongpu(l.batch*l.inputs, 1, l.delta_gpu, 1, state.delta, 1);
}

#endif
