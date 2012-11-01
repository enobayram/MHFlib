/*
 * iau_ukf.hpp
 *
 *  Created on: Nov 17, 2010
 *      Author: enis
 */

#ifndef IAU_UKF_EIGEN_HPP_
#define IAU_UKF_EIGEN_HPP_

#include <iostream>
#include "iau_ukf_eigen.h"

template <int statedim, int procnoisedim, int inputdim, class host>
void iau_ukf::predict(Matrix<double,statedim,1> & stateest, Matrix<double,statedim,statedim> & statecov,
		const Matrix<double,inputdim,1> & input, const Matrix<double,procnoisedim,procnoisedim> & noisecov, host & hostobj,
		Matrix<double,statedim,1> (host::*statetrans)(const Matrix<double,statedim,1> & oldstate, const Matrix<double,procnoisedim,1> & noise, const Matrix<double,inputdim,1> & input) ) {

	const int L = statedim+procnoisedim, kappa = 1, beta=2;
	const double alpha = 1e-3;
	const double lambda = alpha*alpha*(L+kappa)-L;
	const double gamma = lambda + L;


	Matrix<double,statedim,1> statepoints[2*L+1];

	//	Matrix<double,statedim,statedim> statedevs = Matrix<double,statedim,statedim>(gamma*statecov).chol();
	Matrix<double,statedim,statedim> statedevs = (gamma*statecov).llt().matrixL();
	Matrix<double,procnoisedim,procnoisedim> noisedevs = (gamma*noisecov).llt().matrixL();
	Matrix<double,procnoisedim,1> noiseest = Matrix<double,procnoisedim,1>::Zero();

	statepoints[0]=(hostobj.*statetrans)(stateest,noiseest,input);

	for(int i=0; i<statedim; i++) {
		Matrix<double,statedim,1> dev = statedevs.col(i);//(statedevs,0,i);
		statepoints[i+1] = (hostobj.*statetrans)(stateest+dev,noiseest,input);
		statepoints[i+L+1] = (hostobj.*statetrans)(stateest-dev,noiseest,input);
	}

	for(int i=0; i<procnoisedim; i++) {
		Matrix<double,procnoisedim,1> dev = noisedevs.col(i); //(noisedevs,0,i);
		statepoints[i+statedim+1] = (hostobj.*statetrans)(stateest,noiseest+dev,input);
		statepoints[i+L+statedim+1] = (hostobj.*statetrans)(stateest,noiseest-dev,input);
	}

	const double Ws0 = lambda/(L+lambda);
	const double Wsi = 1/(2*(L+lambda));
	const double Wc0 = Ws0+(1-alpha*alpha+beta);
	const double Wci = Wsi;

	stateest =  Ws0 * statepoints[0];

	for(int i=1; i<2*L+1; i++ ) {
		stateest += Wsi*statepoints[i];
	}

	statecov = Wc0 * (statepoints[0]-stateest) * Matrix<double,statedim,1>(statepoints[0]-stateest).transpose();

	for(int i=1; i<2*L+1; i++ ) {
		statecov += Wci * (statepoints[i]-stateest) * Matrix<double,statedim,1>(statepoints[i]-stateest).transpose();
	}

}

// This function returns the likelihood without the constant term
template <int statedim, int measnoisedim, int measdim, class auxclass, class host>
double iau_ukf::update(Matrix<double,statedim,1> & stateest,
		Matrix<double,statedim,statedim> & statecov,
		const Matrix<double,measdim,1> meas,
		const Matrix<double,measnoisedim,measnoisedim> noisecov,
		auxclass auxin,
		host & hostobj,
		Matrix<double,measdim,1> (host::*measerreq)(Matrix<double,statedim,1> state,
				Matrix<double,measnoisedim,1> noise,
				Matrix<double,measdim,1> meas,
				auxclass auxin)
) {

	const int L = statedim+measnoisedim, kappa = 1, beta=2;
	const double alpha = 1e-3;
	const double lambda = alpha*alpha*(L+kappa)-L;
	const double gamma = lambda + L;


	Matrix<double,measdim,1> measerrpoints[2*L+1];

	Matrix<double,statedim,statedim> statedevs = (gamma*statecov).llt().matrixL();
	Matrix<double,measnoisedim,measnoisedim> noisedevs = (gamma*noisecov).llt().matrixL();
	Matrix<double,measnoisedim,1> noiseest = Matrix<double,measnoisedim,1>::Zero();

	measerrpoints[0]=(hostobj.*measerreq)(stateest,noiseest,meas, auxin);

	for(int i=0; i<statedim; i++) {
		Matrix<double,statedim,1> dev = statedevs.col(i);
		measerrpoints[i+1] = (hostobj.*measerreq)(stateest+dev,noiseest, meas,auxin);
		measerrpoints[i+L+1] = (hostobj.*measerreq)(stateest-dev,noiseest, meas,auxin);
	}

	for(int i=0; i<measnoisedim; i++) {
		Matrix<double,measnoisedim,1> dev = noisedevs.col(i);
		measerrpoints[i+statedim+1] = (hostobj.*measerreq)(stateest,noiseest+dev, meas, auxin);
		measerrpoints[i+L+statedim+1] = (hostobj.*measerreq)(stateest,noiseest-dev, meas, auxin);
	}
	const double Ws0 = lambda/(L+lambda);
	const double Wsi = 1/(2*(L+lambda));
	const double Wc0 = Ws0+(1-alpha*alpha+beta);
	const double Wci = Wsi;

	Matrix<double,measdim,1> measerrest =  Ws0 * measerrpoints[0];

	for(int i=1; i<2*L+1; i++ ) {
		measerrest += Wsi*measerrpoints[i];
	}

	Matrix<double,measdim,measdim> measerrcov = Wc0 * (measerrest-measerrpoints[0]) * Matrix<double,measdim,1>(measerrest-measerrpoints[0]).transpose();
	for(int i=1; i<2*L+1; i++ ) {
		measerrcov += Wci * (measerrest-measerrpoints[i]) * Matrix<double,measdim,1>(measerrest-measerrpoints[i]).transpose();
	}

	Matrix<double,statedim,measdim> statemeascov = Matrix<double, statedim, measdim>::Zero(); // The first element of the sum is 0
	for(int i=0; i<statedim; i++) {
		Matrix<double,statedim,1> dev = statedevs.col(i);
		statemeascov += Wci * (dev) * Matrix<double,measdim,1>(measerrest-measerrpoints[i+1]).transpose();
		statemeascov += -Wci * (dev) * Matrix<double,measdim,1>(measerrest-measerrpoints[i+L+1]).transpose(); // Only the sigma points where the states deviate will contribute
	}

	Matrix<double,statedim,measdim> K = statemeascov*measerrcov.inverse();
	stateest+= K*measerrest;
	statecov-= K*measerrcov*K.transpose();

	PartialPivLU<Matrix<double,measdim,measdim> > errcovlu(measerrcov);

	double exponent = -1.0/2 * double(measerrest.transpose()*errcovlu.solve(measerrest));
	double determinant = errcovlu.determinant();
	double errlikelihood = 1/sqrt(determinant) * exp(exponent);

	return errlikelihood;

}

template <int statedim, int measnoisedim, int measdim, class auxclass>
void iau_ukf::update(Matrix<double,statedim,1> & stateest, Matrix<double,statedim,statedim> & statecov,
		const Matrix<double,measdim,1> meas, const Matrix<double,measnoisedim,measnoisedim> noisecov, auxclass auxin,
		Matrix<double,measdim,1> measeq(Matrix<double,statedim,1> state, Matrix<double,measnoisedim,1> noise, auxclass auxin) ) {
	standardwrapper<statedim,measnoisedim,measdim,auxclass> wrapper = standardwrapper<statedim,measnoisedim,measdim,auxclass>(measeq);
	update(stateest,statecov,meas,noisecov,auxin,wrapper,
			&standardwrapper<statedim,measnoisedim,measdim,auxclass>::measerreq);

}

template <int statedim, int measnoisedim, int measdim, class auxclass>
void iau_ukf::update(Matrix<double,statedim,1> & stateest,
		Matrix<double,statedim,statedim> & statecov,
		const Matrix<double,measdim,1> meas,
		const Matrix<double,measnoisedim,measnoisedim> noisecov,
		auxclass auxin,
		Matrix<double,measdim,1> measerreq(Matrix<double,statedim,1> state,
				Matrix<double,measnoisedim,1> noise,
				Matrix<double,measdim,1> meas,
				auxclass auxin)
) {

	update(stateest,statecov,meas,noisecov,auxin,errwrapper<statedim,measnoisedim,measdim,auxclass>(measerreq),
			&errwrapper<statedim,measnoisedim,measdim,auxclass>::measerreq);
}

template<int indim, int noisedim, int outdim, class auxclass, class host>
void iau_ukf::unscentedtransform(Matrix<double,indim,1> & inest,
		Matrix<double,indim,indim> & incov,
		Matrix<double,noisedim,noisedim> & noisecov,
		Matrix<double,outdim,1> &outest,
		Matrix<double,outdim,outdim> &outcov,
		auxclass auxin,
		host & hostobj,
		Matrix<double,outdim,1> (host::*transformation)(Matrix<double,indim,1> input,
				Matrix<double,noisedim,1> noise,
				auxclass auxin)
) {
	const int L = indim + noisedim, kappa = 1, beta=2;
	const double alpha = 1e-3;
	const double lambda = alpha*alpha*(L+kappa)-L;
	const double gamma = lambda + L;


	Matrix<double,outdim,1> sigmapoints[2*L+1];

	Matrix<double,indim,indim> devs = (gamma*incov).llt().matrixL();
	Matrix<double,noisedim,noisedim> noisedevs = (gamma*noisecov).llt().matrixL();
	Matrix<double,noisedim,1> noiseest = Matrix<double,noisedim,1>::Zero();

	sigmapoints[0]=(hostobj.*transformation)(inest,auxin);

	for(int i=0; i<indim; i++) {
		Matrix<double,indim,1> dev = devs.col(i);
		sigmapoints[i+1] = (hostobj.*transformation)(inest+dev, noiseest, auxin);
		sigmapoints[i+L+1] = (hostobj.*transformation)(inest-dev, noiseest, auxin);
	}

	for(int i=0; i<noisedim; i++) {
		Matrix<double,noisedim,1> dev = noisedevs.col(i);
		sigmapoints[i+indim+1] = (hostobj.*transformation)(inest,noiseest+dev, auxin);
		sigmapoints[i+L+indim+1] = (hostobj.*transformation)(inest,noiseest-dev, auxin);
	}


	const double Ws0 = lambda/(L+lambda);
	const double Wsi = 1/(2*(L+lambda));
	const double Wc0 = Ws0+(1-alpha*alpha+beta);
	const double Wci = Wsi;

	outest =  Ws0 * sigmapoints[0];

	for(int i=1; i<2*L+1; i++ ) {
		outest += Wsi*sigmapoints[i];
	}

	outcov = Wc0 * (sigmapoints[0]-outest) * Matrix<double,outdim,1>(sigmapoints[0]-outest).transpose();

	for(int i=1; i<2*L+1; i++ ) {
		outcov += Wci * (sigmapoints[i]-outest) * Matrix<double,outdim,1>(sigmapoints[i]-outest).transpose();
	}

}

template<int indim, int outdim, class auxclass, class host>
void iau_ukf::unscentedtransform(Matrix<double,indim,1> & inest,
		Matrix<double,indim,indim> & incov,
		Matrix<double,outdim,1> &outest,
		Matrix<double,outdim,outdim> &outcov,
		auxclass auxin,
		host & hostobj,
		Matrix<double,outdim,1> (host::*transformation)(Matrix<double,indim,1>,auxclass)
) {
	const int L = indim, kappa = 1, beta=2;
	const double alpha = 1e-3;
	const double lambda = alpha*alpha*(L+kappa)-L;
	const double gamma = lambda + L;


	Matrix<double,outdim,1> sigmapoints[2*L+1];

	Matrix<double,indim,indim> devs = (gamma*incov).llt().matrixL();

	sigmapoints[0]=(hostobj.*transformation)(inest,auxin);

	for(int i=0; i<L; i++) {
		Matrix<double,indim,1> dev = devs.col(i);
		sigmapoints[i+1] = (hostobj.*transformation)(inest+dev,auxin);
		sigmapoints[i+L+1] = (hostobj.*transformation)(inest-dev,auxin);
	}

	const double Ws0 = lambda/(L+lambda);
	const double Wsi = 1/(2*(L+lambda));
	const double Wc0 = Ws0+(1-alpha*alpha+beta);
	const double Wci = Wsi;

	outest =  Ws0 * sigmapoints[0];

	for(int i=1; i<2*L+1; i++ ) {
		outest += Wsi*sigmapoints[i];
	}

	outcov = Wc0 * (sigmapoints[0]-outest) * Matrix<double,outdim,1>(sigmapoints[0]-outest).transpose();

	for(int i=1; i<2*L+1; i++ ) {
		outcov += Wci * (sigmapoints[i]-outest) * Matrix<double,outdim,1>(sigmapoints[i]-outest).transpose();
	}

}

template<int indim, int outdim, class auxclass>
void iau_ukf::unscentedtransform(Matrix<double,indim,1> & inest, Matrix<double,indim,indim> & incov, Matrix<double,outdim,1> &outest, Matrix<double,outdim,outdim> &outcov,
		auxclass auxin, Matrix<double,outdim,1> (*transformation)(Matrix<double,indim,1>,auxclass)) {
	UTfunctionWrapper<indim,outdim,auxclass> wrapper = UTfunctionWrapper<indim,outdim,auxclass>(transformation);
	unscentedtransform(inest,incov,outest,outcov,auxin,wrapper,
			& UTfunctionWrapper<indim,outdim,auxclass>::apply);
}

#endif /* IAU_UKF_EIGEN_HPP_ */
