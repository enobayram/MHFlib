/*
 * iau_ukf_eigen.h
 *
 *  Created on: Mar 13, 2012
 *      Author: eba
 */

#ifndef IAU_UKF_EIGEN_H_
#define IAU_UKF_EIGEN_H_

#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Cholesky>
#include <eigen3/Eigen/LU>
using namespace Eigen;
class iau_ukf {

	template <int statedim, int measnoisedim, int measdim, class auxclass>
	class standardwrapper {
	public:
		Matrix<double,measdim,1> (*measeq)(Matrix<double,statedim,1> state, Matrix<double,measnoisedim,1> noise, auxclass auxin);
		standardwrapper(Matrix<double,measdim,1> measeq(Matrix<double,statedim,1> state, Matrix<double,measnoisedim,1> noise, auxclass auxin)):measeq(measeq) {}
		Matrix<double,measdim,1> measerreq(Matrix<double,statedim,1> state, Matrix<double,measnoisedim,1> noise, Matrix<double,measdim,1> meas, auxclass auxin) {
			return meas-measeq(state,noise,auxin);
		}
	};

	template <int statedim, int measnoisedim, int measdim, class auxclass>
	class errwrapper {
	public:
		Matrix<double,measdim,1> (*measerroreq)(Matrix<double,statedim,1> state, Matrix<double,measnoisedim,1> noise, Matrix<double,measdim,1> meas, auxclass auxin);
		errwrapper(Matrix<double,measdim,1> measerreq(Matrix<double,statedim,1> state, Matrix<double,measnoisedim,1> noise, Matrix<double,measdim,1> meas, auxclass auxin)):measerroreq(measerreq) {}
		Matrix<double,measdim,1> measerreq(Matrix<double,statedim,1> state, Matrix<double,measnoisedim,1> noise, Matrix<double,measdim,1> meas, auxclass auxin) {
			return measerroreq(state,noise,meas,auxin);
		}
	};

public:

	template <int statedim, int procnoisedim, int inputdim, class host>
	static void predict(Matrix<double,statedim,1> & stateest, Matrix<double,statedim,statedim> & statecov,
			const Matrix<double,inputdim,1> & input, const Matrix<double,procnoisedim,procnoisedim> & noisecov, host & hostobj,
			Matrix<double,statedim,1> (host::*statetrans)(const Matrix<double,statedim,1> & oldstate, const Matrix<double,procnoisedim,1> & noise, const Matrix<double,inputdim,1> & input) );

	// This function returns the likelihood without the constant term
	template <int statedim, int measnoisedim, int measdim, class auxclass, class host>
	static double update(Matrix<double,statedim,1> & stateest,
			Matrix<double,statedim,statedim> & statecov,
			const Matrix<double,measdim,1> meas,
			const Matrix<double,measnoisedim,measnoisedim> noisecov,
			auxclass auxin,
			host & hostobj,
			Matrix<double,measdim,1> (host::*measerreq)(Matrix<double,statedim,1> state,
					Matrix<double,measnoisedim,1> noise,
					Matrix<double,measdim,1> meas,
					auxclass auxin)
	);

	template <int statedim, int measnoisedim, int measdim, class auxclass>
	static void update(Matrix<double,statedim,1> & stateest, Matrix<double,statedim,statedim> & statecov,
			const Matrix<double,measdim,1> meas, const Matrix<double,measnoisedim,measnoisedim> noisecov, auxclass auxin,
			Matrix<double,measdim,1> measeq(Matrix<double,statedim,1> state, Matrix<double,measnoisedim,1> noise, auxclass auxin) );

	template <int statedim, int measnoisedim, int measdim, class auxclass>
	static void update(Matrix<double,statedim,1> & stateest,
			Matrix<double,statedim,statedim> & statecov,
			const Matrix<double,measdim,1> meas,
			const Matrix<double,measnoisedim,measnoisedim> noisecov,
			auxclass auxin,
			Matrix<double,measdim,1> measerreq(Matrix<double,statedim,1> state,
					Matrix<double,measnoisedim,1> noise,
					Matrix<double,measdim,1> meas,
					auxclass auxin)
	);

	template<int indim, int noisedim, int outdim, class auxclass, class host>
	static void unscentedtransform(Matrix<double,indim,1> & inest,
			Matrix<double,indim,indim> & incov,
			Matrix<double,noisedim,noisedim> & noisecov,
			Matrix<double,outdim,1> &outest,
			Matrix<double,outdim,outdim> &outcov,
			auxclass auxin,
			host & hostobj,
			Matrix<double,outdim,1> (host::*transformation)(Matrix<double,indim,1> input,
					Matrix<double,noisedim,1> noise,
					auxclass auxin)
	);

	template<int indim, int outdim, class auxclass, class host>
	static void unscentedtransform(Matrix<double,indim,1> & inest,
			Matrix<double,indim,indim> & incov,
			Matrix<double,outdim,1> &outest,
			Matrix<double,outdim,outdim> &outcov,
			auxclass auxin,
			host & hostobj,
			Matrix<double,outdim,1> (host::*transformation)(Matrix<double,indim,1>,auxclass)
	);

	template<int indim, int outdim, class auxclass>
	class UTfunctionWrapper {
	public:
		Matrix<double,outdim,1> (*transformation)(Matrix<double,indim,1>,auxclass);
		UTfunctionWrapper(Matrix<double,outdim,1> (*transformation)(Matrix<double,indim,1>,auxclass)): transformation(transformation) {}
		Matrix<double,outdim,1> apply(Matrix<double,indim,1> in ,auxclass auxin) {
			return transformation(in,auxin);
		}
	};

	template<int indim, int outdim, class auxclass>
	static void unscentedtransform(Matrix<double,indim,1> & inest, Matrix<double,indim,indim> & incov, Matrix<double,outdim,1> &outest, Matrix<double,outdim,outdim> &outcov,
			auxclass auxin, Matrix<double,outdim,1> (*transformation)(Matrix<double,indim,1>,auxclass));

};

#endif /* IAU_UKF_EIGEN_H_ */
