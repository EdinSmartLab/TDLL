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
#include "Activations.h"

#ifdef USE_MKL
#include "mkl_cblas.h"
#else
#ifndef __STDC_VERSION__ //it should never be defined with g++
#define __STDC_VERSION__ 0
#endif
#include "cblas.h"
#endif

struct Layer
{
  const int size, ID;

  Layer(const int _size, const int _ID) : size(_size), ID(_ID) {}
  virtual ~Layer() {}

  virtual void forward(const std::vector<Activation*>& act,
                       const std::vector<Params*>& param) const=0;

  virtual void bckward(const std::vector<Activation*>& act,
                       const std::vector<Params*>& param,
                       const std::vector<Params*>& grad) const=0;

  virtual void init(std::mt19937& G, const std::vector<Params*>& P) const = 0;

  Activation* allocateActivation(const unsigned batchSize) {
    return new Activation(batchSize, size);
  }
  virtual Params* allocate_params() const = 0;


  virtual void    save(const std::vector<Params*>& param) const {
    if(param[ID] not_eq nullptr) param[ID]->save(std::to_string(ID));
  };

  virtual void restart(const std::vector<Params*>& param) const {
    if(param[ID] not_eq nullptr) param[ID]->save(std::to_string(ID));
  };
};

template<int nOutputs>
struct Input_Layer: public Layer
{
  Input_Layer() : Layer(nOutputs, 0) {
    printf("(%d) Input Layer of sizes Output:%d\n", ID, nOutputs);
  }

  Params* allocate_params() const override {
    // non linear activation layers have no parameters:
    return nullptr;
  }

  void forward(const std::vector<Activation*>& act,
               const std::vector<Params*>& param) const override {}

  void bckward(const std::vector<Activation*>& act,
               const std::vector<Params*>& param,
               const std::vector<Params*>& grad)  const override {}

  void init(std::mt19937& G, const std::vector<Params*>& P) const override {}
};
