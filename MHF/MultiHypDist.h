/*
 * MultiHypDist.h
 *
 *  Created on: Nov 27, 2011
 *      Author: eba
 */

#ifndef MULTIHYPDIST_H_
#define MULTIHYPDIST_H_

#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/SVD>
#include <boost/intrusive/slist.hpp>
#include <vector>
#include "GaussianHypothesis.h"
#include "SplitTable.h"
#include <boost/tuple/tuple.hpp>
#include <boost/foreach.hpp>
#include "iau_ukf_eigen.h"
#include <iostream>
#include <algorithm>

using std::vector;

template<int dim>
class MultiHypDist {
	typedef Matrix<double,dim,dim> CovMatrix;
	typedef Matrix<double,dim,1>   MeanMatrix;
	typedef DiagonalMatrix<double,dim>   VarTransMatrix;

public:
	typename GaussianHypothesis<dim>::list GHlist;
	MultiHypDist(){}
	void static split(const GaussianHypothesis<dim> & gh, typename GaussianHypothesis<dim>::list & result, MeanMatrix & splitCov, SplitTable<1> & table);
	void static uniformSplit(const MeanMatrix & mean, const MeanMatrix & widths, typename GaussianHypothesis<dim>::list & result, MeanMatrix & splitCov, SplitTable<1> & table);
	template <int procnoisedim, int inputdim, class host>
	void predict(
			const Matrix<double,inputdim,1> & input,
			const Matrix<double,procnoisedim,procnoisedim> & noisecov,
			host & hostobj,
			Matrix<double,dim,1> (host::*statetrans)(const Matrix<double,dim,1> & oldstate, const Matrix<double,procnoisedim,1> & noise, const Matrix<double,inputdim,1> & input)
			) {
		BOOST_FOREACH(GaussianHypothesis<dim> & gh, GHlist) {
			iau_ukf::predict(gh.mean,gh.cov,input,noisecov,hostobj,statetrans);
		}
	}

	void split(MeanMatrix & splitCov, SplitTable<1> & table) {
		GHList<dim> newList;
		BOOST_FOREACH(GaussianHypothesis<dim> & gh, GHlist) {
			bool toSplit=false;
			for(int i=0; i<dim; i++)
				if(gh.cov(i,i)<splitCov(i))
					toSplit=true;
			if(toSplit) {
				split(gh,newList,splitCov,table);
			} else {
				newList.push_front(*(new GaussianHypothesis<dim>(gh)));
			}
		}
		clear();
		GHlist.splice_after(GHlist.begin(),newList);
	}

	template <int noisedim, int measdim, class host, class auxclass>
	void update(
			const Matrix<double,measdim,1> & meas,
			const Matrix<double,noisedim,noisedim> & noisecov,
			auxclass auxin,
			host & hostobj,
		    Matrix<double,measdim,1> (host::*measerreq)(Matrix<double,dim,1> state,
													    Matrix<double,noisedim,1> noise,
			 										    Matrix<double,measdim,1> meas,
													    auxclass auxin)
		) {
		double totalWeight = 0;
		BOOST_FOREACH(GaussianHypothesis<dim> & gh, GHlist) {
			double likelihood = iau_ukf::update(gh.mean,gh.cov,meas,noisecov,auxin,hostobj,measerreq);
			gh.weight *= likelihood;
			totalWeight += gh.weight;
		}
		BOOST_FOREACH(GaussianHypothesis<dim> & gh, GHlist) {
			gh.weight /= totalWeight;
		}
	}
	MeanMatrix getMean() {
		MeanMatrix result = MeanMatrix::Zero();
		BOOST_FOREACH(GaussianHypothesis<dim> &gh, GHlist) {result+=gh.mean*gh.weight;}
		return result;
	}

	void normalize() {
		double totalWeight = 0;
		BOOST_FOREACH(GaussianHypothesis<dim> & gh, GHlist) {
			totalWeight += gh.weight;
		}
		BOOST_FOREACH(GaussianHypothesis<dim> & gh, GHlist) {
			gh.weight /= totalWeight;
		}
	}

	CovMatrix getCovariance() {
		CovMatrix result = CovMatrix::Zero();
		MeanMatrix mean = getMean();
		BOOST_FOREACH(GaussianHypothesis<dim> & gh, GHlist) {
			result += gh.cov*gh.weight;
			result += (gh.mean-mean)*(gh.mean-mean).transpose()*gh.weight;
		}
		return result;
	}

	void clear() {
		GHlist.clear_and_dispose(boost::checked_delete<GaussianHypothesis<dim> >);
	}

	void resample(int targetSize) {
		vector<GaussianHypothesis<dim> *> samples(GHlist.size());
		int index=0;
		BOOST_FOREACH(GaussianHypothesis<dim> & gh, GHlist) {
			samples[index] = &gh;
			index++;
		}
		random_shuffle ( samples.begin(), samples.end() );
		double sampleWeight = 1.0/targetSize;
		vector<GaussianHypothesis<dim> *> newsamples(targetSize);
		double weight = (double(rand()) / double(RAND_MAX)) * sampleWeight;
		int sampleNum = 0;
		BOOST_FOREACH(GaussianHypothesis<dim> *gh, samples) {
			while(gh->weight>=weight) {
				newsamples[sampleNum] = gh;
				weight += sampleWeight;
				sampleNum++;
			}
			weight -= gh->weight;
		}
		if(sampleNum!=targetSize) std::cerr<<"Error in resampling MHD!\n";
		GHList<dim> oldList;
		oldList.splice_after(oldList.begin(),GHlist);
		BOOST_FOREACH(GaussianHypothesis<dim> *gh, newsamples) {
		 GHlist.push_front(*(new GaussianHypothesis<dim>(gh->mean,gh->cov,sampleWeight)));
		}
		oldList.clear_and_dispose(boost::checked_delete<GaussianHypothesis<dim> >);
	}

	virtual ~MultiHypDist() {}
};

template<int dim>
std::ostream & operator<<(std::ostream & os, const MultiHypDist<dim> & mhd) {
	os<<mhd.GHlist.size()<<"\n";
	BOOST_FOREACH(const GaussianHypothesis<dim> & gh, mhd.GHlist) {
		os<<gh;
	}
	return os;
}

template<int dim>
std::istream & operator>>(std::istream &is, MultiHypDist<dim> & mhd) {
	mhd.clear();
	int noOfHyps;
	is>>noOfHyps;
	for(int i=0; i<noOfHyps; i++) {
		GaussianHypothesis<dim>  *gh = new GaussianHypothesis<dim>();
		is>>*gh;
		mhd.GHlist.push_front(*gh);
	}
	return is;
}
#endif /* MULTIHYPDIST_H_ */
