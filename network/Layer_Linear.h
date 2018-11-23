//
//  High Performance Computing for Science and Engineering (HPCSE) 2018
//  TDLL: Tiny Deep Learning Library - solution code for exercises 6 and 7.
//
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@gmail.com).
//

#pragma once
#include "Layers.h"

template<int nOutputs, int nInputs>
struct LinearLayer: public Layer
{
  Params* allocate_params() const override {
    // Allocate params: weight of size nInputs*nOutputs, bias of size nOutputs
    return new Params(nInputs*nOutputs, nOutputs);
  }

  LinearLayer(const int _ID) : Layer(nOutputs, _ID)
  {
    printf("(%d) Linear Layer of Input:%d Output:%d\n", ID, nInputs, nOutputs);
    assert(nOutputs>0 && nInputs>0);
  }

  void forward(const std::vector<Activation*>& act,
               const std::vector<Params*>& param) const override
  {
    const int batchSize = act[ID]->batchSize;
    {
            Real*const __restrict__ O = act[ID]->output; //batchSize * nOutputs
      const Real*const __restrict__ B = param[ID]->biases;
      #pragma omp parallel for schedule(static)
      for(int b=0; b<batchSize; b++) std::copy(B, B + nOutputs, O + b*nOutputs);
    }
    gemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
        batchSize, nOutputs, nInputs,
        (Real)1.0, act[ID-1]->output, nInputs,
                   param[ID]->weights, nOutputs,
        (Real)1.0, act[ID]->output, nOutputs);
  }


  void bckward(const std::vector<Activation*>& act,
               const std::vector<Params*>& param,
               const std::vector<Params*>& grad)  const override
  {
    // At this point, act[ID]->dError_dOutput contins derivative of error
    // with respect to the outputs of the network.
    const int batchSize = act[ID]->batchSize;
    { // BackProp to compute bias gradient: dError / dBias
      const Real* const __restrict__ deltas = act[ID]->dError_dOutput;
      Real* const __restrict__ grad_B = grad[ID]->biases; // size nOutputs
      std::fill(grad_B, grad_B + nOutputs, 0);
      #pragma omp parallel for schedule(static, 64/sizeof(Real))
      for(int n=0; n<nOutputs; n++)
        for(int b=0; b<batchSize; b++) grad_B[n] += deltas[n + b*nOutputs];
    }
    { // BackProp to compute weight gradient: dError / dWeights
      gemm(CblasRowMajor, CblasTrans, CblasNoTrans,
          nInputs, nOutputs, batchSize,
          (Real)1.0, act[ID-1]->output, nInputs,
                     act[ID]->dError_dOutput, nOutputs,
          (Real)0.0, grad[ID]->weights, nOutputs);
    }
    { // BackProp to compute dEdO of previous layer
      gemm(CblasRowMajor, CblasNoTrans, CblasTrans,
          batchSize, nInputs, nOutputs,
          (Real)1.0, act[ID]->dError_dOutput, nOutputs,
                     param[ID]->weights, nOutputs,
          (Real)0.0, act[ID-1]->dError_dOutput, nInputs);
    }
  }

  void init(std::mt19937& gen, const std::vector<Params*>& param) const override
  {
    assert(param[ID] not_eq nullptr);
    // get pointers to layer's weights and bias
    Real *const W = param[ID]->weights, *const B = param[ID]->biases;
    assert(param[ID]->nWeights == nInputs*size && param[ID]->nBiases == size);

    // initialize weights with Xavier initialization
    const Real scale = std::sqrt( 6.0 / (nInputs + size) );
    std::uniform_real_distribution<Real> dis(-scale, scale);
    std::generate( W, W + nInputs*nOutputs, [&]() { return dis( gen ); } );
    std::generate( B, B + nOutputs, [&]() { return dis( gen ); } );
  }
};
