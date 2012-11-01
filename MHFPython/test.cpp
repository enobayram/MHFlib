/*
 * test.cpp
 *
 *  Created on: Jan 24, 2012
 *      Author: eba
 */

#include "mhfpython.h"
#include <GaussianHypothesis.h>
#include "visualization.h"
#include "transformations.h"

int main() {
	Matrix<double, dim, 1> mean = Matrix<double,dim,1>::Zero();
	Matrix<double, dim, dim> cov = Matrix<double,dim,dim>::Identity();
	double weight = 1;
	GaussianHypothesis<dim> original(mean,cov,weight);

	samplevector samples = drawSamples(original,100000);
	std::cout<<"samples drawn\n";

	odometryTransformation t = odometryTransformation(0.23,0.001,0);
	Matrix<double, inputdim, 1> inp; inp<<1,1;
	Matrix<double, procnoisedim,procnoisedim> noisecov; noisecov<<1,0,0,1;

	propagateSamples(samples,t,inp,noisecov);
	std::cout<<"samples propagated\n";


	Matrix<double,2,1> pos; pos<<1,1;

	beaconMeasurement m = beaconMeasurement(pos);

	Matrix<double,1,1> meas; meas<<1;
	Matrix<double,1,1> measnoisecov; measnoisecov <<0.01;

	updateSamples(samples,m,meas,measnoisecov);
	std::cout<<"samples updated\n";

	resample(samples);
	std::cout<<"resampled\n";

	resample(samples);
	std::cout<<"finished!";
	return 0;
}
