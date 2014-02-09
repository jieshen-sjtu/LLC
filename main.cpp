/*
 * main.cpp
 *
 *  Created on: Feb 3, 2014
 *      Author: jieshen
 */

#include "LLC_Encoder.hpp"
#include <cstring>

#include <string>
#include <iostream>
#include <fstream>
#include <boost/shared_ptr.hpp>

#include <mkl.h>
using namespace std;
using boost::shared_ptr;

int main(int argc, char* argv[])
{
  // load the codebook
  const uint32_t num_center = 4000;
  const uint32_t dim = 128;
  const uint32_t len_cb = num_center * dim;
  float* codebook = (float*) malloc(sizeof(float) * len_cb);
  memset(codebook, 0, sizeof(float) * len_cb);

  const string bookfile("codebook.txt");
  const string ver_book("ver_codebook.txt");
  ifstream input(bookfile.c_str());
  ofstream output(ver_book.c_str());
  for (uint32_t i = 0; i < len_cb; ++i)
  {
    input >> codebook[i];
    output << codebook[i] << " ";
    if ((i + 1) % dim == 0)
      output << endl;
  }
  input.close();
  output.close();

  cerr << "loaded codebook" << endl;

  // load the testing feature
  const uint32_t num_samples = 16116;
  const uint32_t len_sp = num_samples * dim;
  float* features = (float*) malloc(sizeof(float) * len_sp);
  memset(features, 0, sizeof(float) * len_sp);

  const string featfile("feature.txt");
  const string ver_feat("ver_feature.txt");
  input.open(featfile.c_str());
  output.open(ver_feat.c_str());
  for (uint32_t i = 0; i < len_sp; ++i)
  {
    input >> features[i];
    output << features[i] << " ";
    if ((i + 1) % dim == 0)
      output << endl;
  }
  input.close();
  output.close();
  cerr << "loaded samples" << endl;

  // coding
  shared_ptr<float> base(codebook);
  shared_ptr<float> X(features);
  shared_ptr<float> code;

  jieshen::LLC_Encoder llc_model;
  llc_model.set_base(base, dim, num_center);
  llc_model.SetUp();

  cerr << "init model done" << endl;

  llc_model.Encoder(X, dim, num_samples, code);

  cerr << "encode done" << endl;

  const float* pcode = code.get();

  output.open("code.txt");
  for (uint32_t i = 0; i < num_center; ++i)
    output << pcode[i] << "\n";
  output.close();

}

