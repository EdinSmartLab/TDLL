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

#include "Utils.h"

struct Activation
{
  const int batchSize, layersSize;
  //matrix of size batchSize * layersSize with layer outputs:
  Real* const output;
  //matrix of same size containing:
  Real* const dError_dOutput;

  Activation(const int bs, const int ls) : batchSize(bs), layersSize(ls),
    output(_myalloc(bs*ls)), dError_dOutput(_myalloc(bs*ls))
  {
    clearErrors();
    clearOutput();
    assert(batchSize>0 && layersSize>0);
  }

  ~Activation() { _myfree(output); _myfree(dError_dOutput); }

  inline void clearOutput() {
    memset(output,         0, batchSize*layersSize*sizeof(Real));
  }
  inline void clearErrors() {
    memset(dError_dOutput, 0, batchSize*layersSize*sizeof(Real));
  }
};

struct Params
{
  const int nWeights, nBiases;
  Real* const weights; // size is nWeights
  Real* const biases;  // size is nBiases

  Params(const int _nW, const int _nB): nWeights(_nW), nBiases(_nB),
    weights(_myalloc(_nW)), biases(_myalloc(_nB))
  {
    clearBias();
    clearWeight();
  }

  ~Params() { _myfree(weights); _myfree(biases); }

  inline void clearBias() const {
    memset(biases, 0, nBiases * sizeof(Real) );
  }
  inline void clearWeight() const {
    memset(weights, 0, nWeights * sizeof(Real) );
  }

  inline Real normW() const {
    Real ret = 0;
    for (int i=0; i<nWeights; i++) ret += weights[i];
    return ret;
  }
  inline Real normB() const {
    Real ret = 0;
    for (int i=0; i<nBiases; i++) ret += biases[i];
    return ret;
  }

  void save(const std::string fname) const
  {
    FILE* wFile=fopen(("W_"+fname+".raw").c_str(),"wb");
    FILE* bFile=fopen(("b_"+fname+".raw").c_str(),"wb");
    fwrite(weights, sizeof(Real), nWeights, wFile);
    fwrite(biases,  sizeof(Real),  nBiases, bFile);
    fflush(wFile); fflush(bFile);
    fclose(wFile); fclose(bFile);
  }

  void restart(const std::string fname)
  {
    FILE* wFile=fopen(("W_"+fname+".raw").c_str(),"rb");
    FILE* bFile=fopen(("b_"+fname+".raw").c_str(),"rb");

    size_t wsize = fread(weights, sizeof(Real), nWeights, wFile);
    fclose(wFile);
    if((int)wsize not_eq nWeights){
      printf("Mismatch in restarted weight file %s; container:%lu read:%d. Aborting.\n", fname.c_str(), wsize, nWeights);
      abort();
    }

    size_t bsize = fread(biases, sizeof(Real),  nBiases, bFile);
    fclose(bFile);
    if((int)bsize not_eq nBiases){
      printf("Mismatch in restarted biases file %s; container:%lu read:%d. Aborting.\n", fname.c_str(), bsize, nBiases);
      abort();
    }
  }
};
