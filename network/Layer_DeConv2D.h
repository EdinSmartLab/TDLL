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


template
<
int InX, int InY, int InC, //input image: x:width, y:height, c:color channels
int KnX, int KnY, int KnC, //filter:      x:width, y:height, c:color channels
int OpX, int OpY //output img: x:width, y:height, same color channels as KnC
>
struct Deconv2DLayer: public Layer
{
  Params* allocate_params() const override {
    //number of kernel parameters:
    // 2d kernel size * number of inp channels * number of out channels
    const int nParams = InC * KnY * KnX * KnC;
    const int nBiases = KnC;
    return new Params(nParams, nBiases);
  }

  Deconv2DLayer(const int _ID) : Layer(InY * InX * KnY * KnX * KnC, _ID) {
    static_assert(InX>0 && InY>0 && InC>0, "Invalid input");
    static_assert(KnX>0 && KnY>0 && KnC>0, "Invalid kernel");
    static_assert(OpX>0 && OpY>0, "Invalid outpus");
    print();
  }
  void print() {
    printf("(%d) DeConv: In:[%d %d %d] F:[%d %d %d %d] Out:[%d %d %d %d %d]\n",
           ID, InY,InX,InC, InC,KnY,KnX,KnC, InY,InX,KnY,KnX,KnC);
  }

  void forward(const std::vector<Activation*>& act,
               const std::vector<Params*>& param) const override
  {
    assert(act[ID-1]->layersSize == InY * InX * InC);
    assert(act[ID]->layersSize == InY * InX * KnY * KnX * KnC);
    assert(param[ID]->nWeights == InC * KnY * KnX * KnC);
    assert(param[ID]->nBiases == KnC);

    const int batchSize = act[ID]->batchSize;
    {
      const int mm_outRow = batchSize * InY * InX;
      const int mm_nInner = InC;
      const int mm_outCol = KnY * KnX * KnC;
      // [BS*InY*InX, KnY*KnX*KnC] = [BS*InY*InX, InC] [InC, KnY*KnX*KnC]
      gemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
            mm_outRow, mm_outCol, mm_nInner,
            (Real) 1.0, act[ID-1]->output, mm_nInner,
                        param[ID]->weights, mm_outCol,
            (Real) 0.0, act[ID]->output, mm_outCol
          );
    }
    {
            Real* const __restrict__ O = act[ID]->output;
      const Real* const __restrict__ B = param[ID]->biases; // size is KnC
      #pragma omp parallel for schedule(static)
      for(int i=0; i<batchSize * InY*InX * KnY*KnX*KnC; i++) O[i] += B[i % KnC];
    }
  }

  void bckward(const std::vector<Activation*>& act,
               const std::vector<Params*>& param,
               const std::vector<Params*>& grad) const override
  {
    const int batchSize = act[ID]->batchSize;
    {
      const Real* const __restrict__ D = act[ID]->dError_dOutput;
            Real* const __restrict__ B = grad[ID]->biases; // size is KnC
      std::fill(B, B + KnC, 0);
      #pragma omp parallel for schedule(static) reduction(+ : B[:KnC])
      for(int i=0; i<batchSize * InY*InX * KnY*KnX*KnC; i++) B[i % KnC] += D[i];
    }

    const int mm_outRow = batchSize * InY * InX;
    const int mm_nInner = InC;
    const int mm_outCol = KnY * KnX * KnC;

    // Compute gradient of error wrt to kernel parameters:
    // (  grad  filter  )   (     input     )   (      dErr / dOut      )
    // [InC, KnY*KnX*KnC] = [BS*InY*InX, InC]^T [BS*InY*InX, KnY*KnX*KnC]
    gemm(CblasRowMajor, CblasTrans, CblasNoTrans,
          mm_nInner, mm_outCol, mm_outRow,
          (Real) 1.0, act[ID-1]->output, mm_nInner,
                      act[ID]->dError_dOutput, mm_outCol,
          (Real) 0.0, grad[ID]->weights, mm_outCol);

    // Compute gradient of error wrt to output of previous layer:
    // [BS*InY*InX, InC] = [BS*InY*InX, KnY*KnX*KnC] [InC, KnY*KnX*KnC]^T
    gemm(CblasRowMajor, CblasNoTrans, CblasTrans,
          mm_outRow, mm_nInner, mm_outCol,
          (Real) 1.0, act[ID]->dError_dOutput, mm_outCol,
                      param[ID]->weights, mm_outCol,
          (Real) 0.0, act[ID-1]->dError_dOutput, mm_nInner);
  }

  void init(std::mt19937& gen, const std::vector<Params*>& param) const override
  {
    // get pointers to layer's weights and bias
    Real *const W = param[ID]->weights, *const B = param[ID]->biases;
    // initialize weights with Xavier initialization
    const int nAdded = KnX * KnY * InC, nW = param[ID]->nWeights;
    const Real scale = std::sqrt(6.0 / (nAdded + KnC));
    std::uniform_real_distribution < Real > dis(-scale, scale);
    std::generate(W, W + nW, [&]() {return dis( gen );});
    std::fill(B, B + KnC, 0);
  }
};
