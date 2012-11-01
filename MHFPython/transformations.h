/*
 * transformations.h
 *
 *  Created on: Jan 18, 2012
 *      Author: eba
 */

#ifndef TRANSFORMATIONS_H_
#define TRANSFORMATIONS_H_

#include <eigen3/Eigen/Dense>
#include <boost/math/distributions.hpp>
using namespace Eigen;

class nullclass {};

template<int statedim, int noisedim, int inputdim>
class transformations {
public:
	virtual ~transformations(){}
	virtual Matrix<double,statedim,1> statetrans (
			const Matrix<double,statedim,1> & oldstate,
			const Matrix<double,noisedim,1> & noise,
			const Matrix<double,inputdim,1> & input) {return oldstate;}

	virtual Matrix<double,inputdim,1> measerr(Matrix<double,statedim,1> state,
											  Matrix<double,noisedim,1> noise,
											  Matrix<double,inputdim,1> meas,
											  nullclass auxin){return meas;}

	virtual double measlikelihood(Matrix<double,statedim,1> state,
								  Matrix<double,noisedim,noisedim> noisecov,
								  Matrix<double,inputdim,1> meas,
								  nullclass auxin) {return 1;}
};

class odometryTransformation: public transformations<3,2,2> {
	double B;
	double E2;
	double A2;
public:
	odometryTransformation(double B, double E2, double A2): B(B),E2(E2),A2(A2){}
	Matrix<double,3,1> statetrans(const Matrix<double,3,1> & oldstate,
								  const Matrix<double,2,1> & noise,
								  const Matrix<double,2,1> & input
								  ){
		double lr = input(0) + noise(0) * sqrt(E2*fabs(input(0)));
		double ll = input(1) + noise(1) * sqrt(E2*fabs(input(1)));
		double L = (lr+ll)/2;
		double D = (lr-ll)/(2*B);
		Matrix<double,3,1> newstate; newstate = oldstate;
		Vector2d unitx; unitx << 1,0;
		newstate.block<2,1>(0,0) = oldstate.block<2,1>(0,0) + Rotation2D<double>(oldstate(2)+D/2) * unitx * L;
		newstate(2) = oldstate(2) + D;
		return newstate;
	}
};

class beaconMeasurement: public transformations<3,1,1> {
	Vector2d pos;
	double estimateDist(const Matrix<double,3,1> & state) {
		Vector2d displacement = state.block<2,1>(0,0)-pos;
		return displacement.norm();
	}

public:
	beaconMeasurement(const Vector2d & pos): pos(pos){}
	virtual Matrix<double,1,1> measerr(Matrix<double,3,1> state,
											  Matrix<double,1,1> noise,
											  Matrix<double,1,1> meas,
											  nullclass auxin
	){
		Matrix<double,1,1> error; error << (meas(0) + noise(0) - estimateDist(state));
		return error;
	}
	virtual double measlikelihood(Matrix<double,3,1> state,
								  Matrix<double,1,1> noisecov,
								  Matrix<double,1,1> meas,
								  nullclass auxin
	) {
		boost::math::normal s(0,sqrt(noisecov(0)));
		double estimated = estimateDist(state);
		return boost::math::pdf(s,estimated-meas(0));
	}
};



#endif /* TRANSFORMATIONS_H_ */
