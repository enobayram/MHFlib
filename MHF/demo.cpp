/*
 * main.cpp
 *
 *  Created on: Nov 27, 2011
 *      Author: eba
 */
#include "GaussianHypothesis.h"
#include "MultiHypDist.hpp"
#include <iostream>
#include <eigen3/Eigen/Dense>
#include "SplitTable.h"
#include "iau_ukf_eigen.hpp"

using std::cout;

void testUKF();

int main(int argc, char **argv) {
	GaussianHypothesis<3> h;
	h.mean << 0,0,0;
	h.cov << 5,4.9,0,
			 4.9,5,0,
			 0,0,0.1;
	h.weight=1;
	Matrix<double, 3, 1> variances;
	variances << 1,2,3;
	GaussianHypothesis<3>::list splittedList;
	MultiHypDist<3> mhf;
	SplitTable<1> table;
	cout<<" Table Size:"<<table.table.size()<<"\n";
	mhf.split(variances, table);
	//cout<<table.getUpperEntry(5).variance;
	cout<<splittedList.size()<<"\n";
	BOOST_FOREACH(GaussianHypothesis<3> & g, splittedList) {
		cout<<g<<"\n";
	}
	testUKF();
}

class hostclass {
public:
	Matrix<double,3,1> statetrans(const Matrix<double,3,1> & oldstate, const Matrix<double,2,1> & noise, const Matrix<double,1,1> &input); // state transition equation for the UKF
};

Matrix<double,1,1> measeq(Matrix<double,3,1> state, Matrix<double,1,1> noise, Matrix<double,0,1> auxin); // Measurement equation for the UKF

Matrix<double,1,1> measerreq(Matrix<double,3,1> state, Matrix<double,1,1> noise, Matrix<double,1,1> meas, Matrix<double,0,1> auxin) {
	Matrix<double,1,1> result; result << meas-measeq(state,noise,auxin);
	return result;
}

Matrix<double,2,1> transformation(Matrix<double,3,1> in, int auxin) {
	Matrix<double,2,1> result;
	result(0,0) = in(0,0)+in(1,0);
	result(1,0) = in(0,0)-in(1,0);
	return result;
}

void testUKF() {
	cout<<"starting testUKF\n";
	const int statedim = 3, procnoisedim = 2, inputdim = 1, measdim = 1, measnoisedim = 1;

	Matrix<double,statedim,1> stateest; stateest << 1,
											   2,
											   3;
	Matrix<double,statedim,statedim> statecov; statecov << 4,0,0,
													  0,4,0,
													  0,0,4;
	Matrix<double,procnoisedim,procnoisedim> noisecov; noisecov << 4,0,
															  0,1;
	Matrix<double,inputdim,1> input; input << 5;

	hostclass hostobj;

	cout<<"\nstateest:\n"<<stateest;
	cout<<"\nstatecov:\n"<<statecov;
	cout<<"\nnoisecov:\n"<<noisecov;
	iau_ukf::predict(stateest, statecov,input,noisecov,hostobj,&hostclass::statetrans);
	cout<<"\n=UKF predict=\n";
	cout<<"\nstateest:\n"<<stateest;
	cout<<"\nstatecov:\n"<<statecov;


	Matrix<double,measnoisedim,measnoisedim> measnoisecov; measnoisecov << 4;
	Matrix<double,measdim,1> meas; meas << 11;

	iau_ukf::update(stateest,statecov,meas,measnoisecov,Matrix<double,0,1>(),measeq);

	cout<<"\n=UKF update=\n";
	cout<<"\nstateest:\n"<<stateest;
	cout<<"\nstatecov:\n"<<statecov;

	Matrix<double,3,1> inest; inest << 3,2,1;
	Matrix<double,3,3> incov; incov << 2,0,0,
								  0,1,0,
								  0,0,1;
	Matrix<double,2,1> outest;
	Matrix<double,2,2> outcov;

	iau_ukf::unscentedtransform(inest,incov,outest,outcov,0,transformation);

	cout<<"\n=unscented transform=\n";

	cout<<"\noutest:\n"<<outest;
	cout<<"\noutcov:\n"<<outcov;

}

Matrix<double,3,1> hostclass::statetrans(const Matrix<double,3,1> & oldstate, const Matrix<double,2,1> & noise, const Matrix<double,1,1> &input) {
	Matrix<double,3,1> result; result=oldstate;
	result(0,0) += exp(noise(0,0));
	result(1,0) += noise(0,0);
	result(2,0) += input(0,0);
	return result;
}

Matrix<double,1,1> measeq(Matrix<double,3,1> state, Matrix<double,1,1> noise, Matrix<double,0,1> auxin) {
	Matrix<double,1,1> result; result<<(noise(0,0) + state(0,0) + state(1,0) + state(2,0));
	return result;
}

