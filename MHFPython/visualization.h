/*
 * visualization.h
 *
 *  Created on: Jan 24, 2012
 *      Author: eba
 */

#ifndef VISUALIZATION_H_
#define VISUALIZATION_H_
#include <GaussianHypothesis.h>
#include <eigen3/Eigen/Dense>
#include <vector>
#include <boost/random.hpp>
#include <boost/random/normal_distribution.hpp>
#include <algorithm>
#include <boost/foreach.hpp>
#include <boost/tuple/tuple.hpp>
#include <cstdlib>

#include "transformations.h"


struct Sample {
	Matrix<double,dim,1> state;
	double weight;
	Sample(){}
	Sample(const Matrix<double,dim,1> & state, double weight): state(state), weight(weight){}
	Sample(const Sample & other): state(other.state), weight(other.weight){}
	Sample & operator=(const Sample & other) {state = other.state; weight=other.weight; return *this;}
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};
typedef std::vector<Sample> samplevector;

template<int rows, int cols, class Generator>
Matrix<double,rows,cols> randomMatrix(Generator & generator) {
	Matrix<double, rows, cols> result;
	for(int i=0; i<rows; i++) {
		for (int j = 0; j < cols; ++j) {
			result(i,j) = generator();
		}
	}
	return result;
}

template<int dim>
samplevector drawSamples(const GaussianHypothesis<dim> & gh, int samples) {
    const Matrix<double,dim,1> & mean = gh.mean;
    Matrix<double,dim,dim> cholState = gh.cov.llt().matrixL();

    boost::mt19937 rng;
    boost::normal_distribution<> nd(0,1);
    boost::variate_generator<boost::mt19937&, boost::normal_distribution<> > var_nor(rng, nd);

    samplevector result(samples);

    for (int i=0; i<samples; i++) {
		result[i] = Sample(mean + cholState * randomMatrix<dim,1>(var_nor),1.0/samples);
    }
    return result;
}

void propagateSamples(samplevector & samples,
					  transformations<dim,procnoisedim,inputdim> &trans,
					  const Matrix<double,inputdim,1> & input,
					  const Matrix<double,procnoisedim,procnoisedim> & noisecov);

void updateSamples(samplevector & samples,
				   transformations<dim,measdim,measdim> &trans,
				   const Matrix<double,measdim,1> & meas,
				   const Matrix<double,measdim,measdim> & noisecov);

void resample(samplevector & samples);

#endif /* VISUALIZATION_H_ */
