/*
 * ukf.cpp
 *
 *  Created on: Jan 12, 2012
 *      Author: eba
 */


#include "mhfpython.h"
#include <MultiHypDist.h>
#include <iau_ukf_eigen.hpp>
#include <boost/foreach.hpp>
#include <eigen3/Eigen/Cholesky>
#include <eigen3/Eigen/Geometry>

#include "transformations.h"

template<int dim, int procnoisedim, int inputdim>
struct transformationsWrap : transformations<dim,procnoisedim,inputdim>
							,wrapper<transformations<dim,procnoisedim,inputdim> >
{
	virtual Matrix<double,dim,1> statetrans(
			const Matrix<double,dim,1> & oldstate,
			const Matrix<double,procnoisedim,1> & noise,
			const Matrix<double,inputdim,1> & input) {return this->get_override("statetrans")(oldstate,noise,input);}
	virtual Matrix<double,inputdim,1> measerr(Matrix<double,dim,1> state,
											  Matrix<double,procnoisedim,1> noise,
											  Matrix<double,inputdim,1> meas,
											  nullclass auxin){return this->get_override("measerr")(state,noise,meas);}
};

template< class T > // T extends transformations
void UKFPredict(
		MultiHypDist<dim> & mhd,
		const Matrix<double,inputdim,1> & input,
		const Matrix<double,procnoisedim,procnoisedim> & noisecov,
		T & transobj
		) {
	mhd.predict(input,noisecov,transobj,&T::statetrans);
}

template< class T > // T extends transformations
void UKFUpdate(
		MultiHypDist<dim> & mhd,
		const Matrix<double,measdim,1> & meas,
		const Matrix<double,measdim,measdim> & noisecov,
		T & transobj
		) {
	nullclass nullobj;
	mhd.update(meas,noisecov,nullobj,transobj,&T::measerr);
}

//struct odometryWrap: public odometryTransformation, public wrapper<transformations<3,2,2> >{
//	odometryWrap(double B, double E2, double A2): odometryTransformation(B,E2,A2){}
//};

void export_ukfs() {
	class_<transformationsWrap<dim,procnoisedim,inputdim>, boost::noncopyable>("transformations")
	    ;
	class_<transformationsWrap<dim,measdim,measdim>, boost::noncopyable>("transformationsMeas")
	    ;

    def("UKFPredict",UKFPredict<transformations<dim,procnoisedim,inputdim> >);
    def("UKFUpdate",UKFUpdate<transformations<dim,measdim,measdim> >);
}
