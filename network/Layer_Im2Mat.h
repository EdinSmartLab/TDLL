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

// Im2MatLayer gets as input an image of sizes InX * InY * InC
// and prepares the output for convolution with a filter of size KnY * KnX * KnC
// and output an image of size OpY * OpX * KnC
template
<
  int InX, int InY, int InC, //input image: x:width, y:height, c:color channels
  int KnX, int KnY, int KnC, //filter:      x:width, y:height, c:color channels
  int Sx, int Sy, // stride  x/y
  int Px, int Py, // padding x/y
  int OpX, int OpY //output img: x:width, y:height, same color channels as KnC
>
struct Im2MatLayer: public Layer
{
  // if not transposed then forward operation is Im2Mat and backward is Mat2Im
  // if transposed then forward operation is Mat2Im and backward is Im2Mat
  const bool transposed;
  const int inp_size = transposed ? OpY*OpX*KnY*KnX*InC : InX*InY*InC;
  const int out_size = transposed ? InX*InY*InC : OpY*OpX*KnY*KnX*InC;

  //Im2ColLayer has no parameters:
  Params* allocate_params() const override { return nullptr; }

  Im2MatLayer(const int _ID, const bool bTrans = false) :
    Layer(bTrans? InX*InY*InC : OpY*OpX*KnY*KnX*InC, _ID), transposed(bTrans) {
    static_assert(Sx> 0 && Sy> 0, "Invalid stride");
    static_assert(Px>=0 && Py>=0, "Invalid kernel");
    print();
  }

  void print() {
    if (transposed)
      printf("(%d) Col2Im transform Mat:[%d %d %d %d %d] to Img:[%d %d %d] ",
          ID, OpY,OpX,KnY,KnX,InC, InY,InX,InC);
    else
      printf("(%d) Im2Col transform Img:[%d %d %d] to Mat:[%d %d %d %d %d] ",
          ID, InY,InX,InC, OpY,OpX,KnY,KnX,InC);
    printf("with Stride:[%d %d] and Padding:[%d %d]\n",Sx,Sy,Px,Py);
  }

  void forward(const std::vector<Activation*>& act,
               const std::vector<Params*>& param) const override
  {
    const int batchSize = act[ID]->batchSize;

    if(transposed)
    {
      assert(act[ID-1]->layersSize == OpY * OpX * KnY * KnX * InC);
      assert(act[ID]->layersSize == InX * InY * InC);
      Mat2Im(batchSize, act[ID-1]->output, act[ID]->output);
    }
    else
    {
      assert(act[ID-1]->layersSize == InX * InY * InC);
      assert(act[ID]->layersSize == OpY * OpX * KnY * KnX * InC);
      Im2Mat(batchSize, act[ID-1]->output, act[ID]->output);
    }
  }

  void bckward(const std::vector<Activation*>& act,
               const std::vector<Params*>& param,
               const std::vector<Params*>& grad) const override
  {
    const int batchSize = act[ID]->batchSize;

    if(transposed)
    {
      assert(act[ID-1]->layersSize == OpY * OpX * KnY * KnX * InC);
      assert(act[ID]->layersSize == InX * InY * InC);
      Im2Mat(batchSize, act[ID]->dError_dOutput, act[ID-1]->dError_dOutput);
    }
    else
    {
      assert(act[ID-1]->layersSize == InX * InY * InC);
      assert(act[ID]->layersSize == OpY * OpX * KnY * KnX * InC);
      Mat2Im(batchSize, act[ID]->dError_dOutput, act[ID-1]->dError_dOutput);
    }
  }

  void Im2Mat(const int BS,
    const Real*const __restrict__ lin_inp,
    Real*const __restrict__ lin_out
  ) const
  {
    using InputImages    = Real[][InY][InX][InC];
    using OutputMatrices = Real[][OpY][OpX][KnY][KnX][InC];

    // Convert pointers to a reference to multi dim arrays for easy access:
    // 1) INP is a reference: i'm not creating new data
    // 2) The type of INP is an array of sizes [???][InY][InX][InC]
    // 3) The first dimension is the batchsize and is not known at compile time
    // 4) Because it's the slowest index the compiler does not complain
    // 5) The conversion should be read from right to left: (A) convert lin_inp
    // to pointer to a static multi-array of size [???][InY][InX][InC]
    // (B) Return the reference of the memory space pointed at by a.
    const InputImages & __restrict__ INP = * (InputImages*) lin_inp;
    //                                    (B)(     A      )
    OutputMatrices & __restrict__ OUT = * (OutputMatrices*) lin_out;

    // clean up memory space of lin_out. Why? Because padding, that's why.
    memset(lin_out, 0, BS * OpY * OpX * KnY * KnX * InC * sizeof(Real) );

    #pragma omp parallel for collapse(3) schedule(static)
    for (int bc = 0; bc < BS;  bc++)
    for (int oy = 0; oy < OpY; oy++)
    for (int ox = 0; ox < OpX; ox++)
    {
      //starting position along input map for convolution with kernel
      const int ix0 = ox * Sx - Px, iy0 = oy * Sy - Py;
      for (int fy = 0; fy < KnY; fy++)
      for (int fx = 0; fx < KnX; fx++)
      {
        //index along input map of the convolution op:
        const int ix = ix0 + fx, iy = iy0 + fy;
        //padding: skip addition if outside input boundaries
        if (ix < 0 || ix >= InX || iy < 0 || iy >= InY) continue;
        for (int ic = 0; ic < InC; ic++) //loop over inp feature maps
          OUT[bc][oy][ox][fy][fx][ic] = INP[bc][iy][ix][ic];
      }
    }
  }

  void Mat2Im(const int BS,
    const Real*const __restrict__ lin_inp,
    Real*const __restrict__ lin_out
  ) const
  {
    using InputImages    = Real[][InY][InX][InC];
    using OutputMatrices = Real[][OpY][OpX][KnY][KnX][InC];
    // Output is d Loss d Input, same size as INP before:
    InputImages & __restrict__ dLdINP = * (InputImages*) lin_out;
    // Input is d Loss d Output, same size as OUT before:
    const OutputMatrices & __restrict__ dLdOUT = * (OutputMatrices*) lin_inp;

    // Mat2Im accesses memory with plus equal: reset field
    memset(lin_out, 0, BS * InY * InX * InC * sizeof(Real) );

    #pragma omp parallel for collapse(3) schedule(static)
    for (int bc = 0; bc < BS;  bc++)
    for (int iy = 0; iy < InY; iy++)
    for (int ix = 0; ix < InX; ix++)
    {
      for (int fy = 0; fy < KnY; fy++)
      for (int fx = 0; fx < KnX; fx++)
      {
        const int oy = ( iy + Py - fy ) / Sy;
        const int ox = ( ix + Px - fx ) / Sx;
        //padding: skip addition if outside input boundaries
        if (oy < 0 || oy >= OpX || ox < 0 || ox >= OpY) continue;
        for (int ic = 0; ic < InC; ic++) //loop over inp feature maps
          dLdINP[bc][iy][ix][ic] += dLdOUT[bc][oy][ox][fy][fx][ic];
      }
    }
  }

  void init(std::mt19937& G, const std::vector<Params*>& P) const override {  }
};
