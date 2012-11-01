/*
 * GaussianHypothesis.h
 *
 *  Created on: Nov 27, 2011
 *      Author: eba
 */
#ifndef GAUSSIANHYPOTHESIS_H_
#define GAUSSIANHYPOTHESIS_H_

#include <eigen3/Eigen/Dense>
#include <boost/intrusive/slist.hpp>
#include <iostream>
#include <boost/checked_delete.hpp>

using namespace Eigen;
using boost::intrusive::slist_base_hook;
using boost::intrusive::slist;
using std::cout;

template<int dim> struct GaussianHypothesis;

template<int dim> class GHList: public slist<GaussianHypothesis<dim> > {
public:
	~GHList() {
		clear_and_dispose(boost::checked_delete<GaussianHypothesis<dim> >);
	}
};

template<int dim>
struct GaussianHypothesis: public slist_base_hook<> {
	typedef Matrix<double,dim,dim> CovMatrix;
	typedef Matrix<double,dim,1>   MeanMatrix;

public:
	MeanMatrix mean;
	CovMatrix cov;
	double weight;
	//typedef slist<GaussianHypothesis<dim> > list;
	typedef GHList<dim> list;
	GaussianHypothesis(){};
	GaussianHypothesis(const MeanMatrix & mean, const CovMatrix & cov, double weight): mean(mean), cov(cov), weight(weight){}
	//~GaussianHypothesis() {cout<<"Deleting Gaussian Hypothesis\n";}
	//virtual ~GaussianHypothesis();
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

template<int dim>
std::ostream & operator<<(std::ostream & os, const GaussianHypothesis<dim> & gh) {
	return os<<gh.mean<<"\n"<<gh.cov<<"\n"<<gh.weight<<"\n";
}

template <int dim>
std::istream & operator>>(std::istream &is,GaussianHypothesis<dim> &g)
{
	for(int i=0; i<dim; i++) {
		double & element = g.mean(i,0);
		is >> element;
	}
	for (int i = 0; i < dim; ++i) {
		for (int j = 0; j < dim; ++j) {
			is>>g.cov(i,j);
		}
	}
	return is>>g.weight;
}


#endif /* GAUSSIANHYPOTHESIS_H_ */
