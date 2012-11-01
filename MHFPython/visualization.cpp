/*
 * visualization.cpp
 *
 *  Created on: Jan 18, 2012
 *      Author: eba
 */

#include "mhfpython.h"

#include <boost/foreach.hpp>
#include "visualization.h"
#include <numpy/noprefix.h>


// TODO expand the definition of this function to handle higher dimensional cases
//PyObject * actualDistributionAfterPredict(const GaussianHypothesis<dim> & gh,
//										 transformations<dim,procnoisedim,inputdim> &trans,
//										 const Matrix<double,inputdim,1> & input,
//										 const Matrix<double,procnoisedim,procnoisedim> & noisecov,
//										 int samples, double limit, const int steps) {
//    double * zData = new double[steps * steps];
//    Map<Matrix<double, Dynamic, Dynamic> > z(zData, steps,steps);
//    z.fill(0);
//    double cellArea = pow(2 * limit / steps,2);
//    const Matrix<double,dim,1> & mean = gh.mean;
//    Matrix<double,dim,dim> cholState = gh.cov.llt().matrixL();
//    Matrix<double,procnoisedim,procnoisedim> cholNoise = noisecov.llt().matrixL();
//    boost::mt19937 rng;
//    boost::normal_distribution<> nd(0,1);
//    boost::variate_generator<boost::mt19937&, boost::normal_distribution<> > var_nor(rng, nd);
//
//
//    for (int i=0; i<samples; i++) {
//		Matrix<double,dim,1> sample = mean + cholState * randomMatrix<dim,1>(var_nor);
//		Matrix<double,procnoisedim,1> noise = cholNoise * randomMatrix<procnoisedim,1>(var_nor);
//		sample = trans.statetrans(sample,noise,input);
//		if(limit>sample.maxCoeff() && sample.minCoeff()>-limit) {
//			Matrix<int,dim,1> index = (sample.array() * steps/2 / limit + steps/2).matrix().cast<int>();
//			z(index(0,0),index(1,0)) += 1./(samples*cellArea);
//		}
//	}
//    int N[] = {steps,steps};
//    PyArrayObject * result = (PyArrayObject*) PyArray_SimpleNewFromData(2,N, PyArray_DOUBLE, zData);
//    result->flags = result->flags | OWNDATA;
//    return (PyObject *)result;
//}

void propagateSamples(samplevector & samples,
					  transformations<dim,procnoisedim,inputdim> &trans,
					  const Matrix<double,inputdim,1> & input,
					  const Matrix<double,procnoisedim,procnoisedim> & noisecov
	) {
    Matrix<double,procnoisedim,procnoisedim> cholNoise = noisecov.llt().matrixL();
    boost::mt19937 rng;
    boost::normal_distribution<> nd(0,1);
    boost::variate_generator<boost::mt19937&, boost::normal_distribution<> > var_nor(rng, nd);

    BOOST_FOREACH(Sample & sample, samples) {
		Matrix<double,procnoisedim,1> noise = cholNoise * randomMatrix<procnoisedim,1>(var_nor);
		sample.state = trans.statetrans(sample.state,noise,input);
	}
}

void updateSamples(samplevector & samples,
				   transformations<dim,measdim,measdim> &trans,
				   const Matrix<double,measdim,1> & meas,
				   const Matrix<double,measdim,measdim> & noisecov
	) {

	nullclass nullobj;
	double totalWeight = 0;
    BOOST_FOREACH(Sample & sample, samples) {
		sample.weight = trans.measlikelihood(sample.state,noisecov,meas,nullobj);
		totalWeight += sample.weight;
	}
    BOOST_FOREACH(Sample & sample, samples) {
    	sample.weight /= totalWeight;
    }
}

// TODO expand the definition of this function to handle higher dimensional cases
PyObject * layoutSamples(const samplevector & samples, double limit, const int steps) {
    double * zData = new double[steps * steps];
    Map<Matrix<double, Dynamic, Dynamic> > z(zData, steps,steps);
    z.fill(0);
    double cellArea = pow(2 * limit / steps,2);

    typedef Matrix<double,dim,1> SampleMat;

    BOOST_FOREACH(const Sample & sample, samples) {
		if(limit>sample.state.block<2,1>(0,0).maxCoeff() && sample.state.block<2,1>(0,0).minCoeff()>-limit) { // TODO This line is temporary
			Matrix<int,dim,1> index = (sample.state.array() * steps/2 / limit + steps/2).matrix().cast<int>();
			z(index(0,0),index(1,0)) += sample.weight/cellArea;
		}
	}
    npy_intp N[] = {steps,steps};
    PyArrayObject * result = (PyArrayObject*) PyArray_SimpleNewFromData(2,N, PyArray_DOUBLE, zData);
    result->flags = result->flags | OWNDATA;
    return (PyObject *)result;
}

void resample(samplevector & samples) {
	random_shuffle ( samples.begin(), samples.end() );
	int sampleCount = samples.size();
	double sampleWeight = 1.0/sampleCount;
	samplevector oldsamples = samples;
	double weight = (double(rand()) / double(RAND_MAX)) * sampleWeight;
	int sampleNum = 0;
	BOOST_FOREACH(const Sample & sample, oldsamples) {
		while(sample.weight>=weight) {
			samples[sampleNum].state = sample.state;
			samples[sampleNum].weight = sampleWeight;
			weight += sampleWeight;
			sampleNum++;
		}
		weight -= sample.weight;
	}
	if(sampleNum!=sampleCount) std::cerr<<"problem!\n";
}

void export_visualization() {
	import_array();
	class_<samplevector>("samplevector");
	//def("actualDistributionAfterPredict", actualDistributionAfterPredict);
	def("drawSamples", drawSamples<dim>);
	def("propagateSamples", propagateSamples);
	def("updateSamples", updateSamples);
	def("layoutSamples", layoutSamples);
	def("resample", resample);
}
