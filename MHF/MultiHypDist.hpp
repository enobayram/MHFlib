/*
 * MultiHypDist.hpp
 *
 *  Created on: Mar 13, 2012
 *      Author: eba
 */

#ifndef MULTIHYPDIST_HPP_
#define MULTIHYPDIST_HPP_

#include "MultiHypDist.h"
#include "iau_ukf_eigen.hpp"

template<int dim>
void MultiHypDist<dim>::uniformSplit(const MeanMatrix & mean, const MeanMatrix & widths, typename GaussianHypothesis<dim>::list & result, MeanMatrix & splitCov, SplitTable<1> & table) {
	MeanMatrix standardDevs = splitCov.cwiseSqrt();
	const vector<GaussianHypothesis<1> > * independentHypotheses[dim];
	double scaleRatios[dim];

	for(int i=0; i<dim; i++) {
		double entryWidth = widths(i)/standardDevs(i);
		TableEntry<1> entry = table.getUpperEntry(entryWidth);
		scaleRatios[i] = sqrt(entryWidth/entry.variance);
		independentHypotheses[i] = &entry.hypotheses;
	}

	bool finished = false;
	vector<GaussianHypothesis<1> >::const_iterator iterators[dim];
	for(int i=0; i<dim; i++) iterators[i] = independentHypotheses[i]->begin();
	while(!finished) {
		GaussianHypothesis<dim> * newHypothesis = new GaussianHypothesis<dim>();
		DiagonalMatrix<double,dim> baseCov;
		MeanMatrix baseMean;
		newHypothesis->weight = 1;
		for(int i=0; i<dim; i++) {
			baseMean(i) = iterators[i]->mean(0)*scaleRatios[i]*standardDevs[i];
			baseCov.diagonal()(i) = iterators[i]->cov(0,0)*scaleRatios[i]*scaleRatios[i]*standardDevs[i]*standardDevs[i];
			newHypothesis->weight *= iterators[i]->weight;
		}
		newHypothesis->mean=baseMean;
		newHypothesis->cov=baseCov;
		result.push_front(*newHypothesis);
		for(int i=0; i<dim; i++) { // This loop is "incrementing" the "iterators" vector to the next "state" in an abstract sense
			iterators[i]++;
			if(iterators[i]==independentHypotheses[i]->end()) {
				iterators[i]=independentHypotheses[i]->begin();
				if(i==dim-1) finished = true;
			} else break;
		}
	}
}


template<int dim>
void MultiHypDist<dim>::split(const GaussianHypothesis<dim> & gh, typename GaussianHypothesis<dim>::list & result, MeanMatrix & splitCov, SplitTable<1> & table) {

	VarTransMatrix splitCovTransform; splitCovTransform.diagonal() = splitCov.array().sqrt();
	VarTransMatrix splitCovInvTransform = splitCovTransform.inverse();

	CovMatrix stretchedCov = splitCovInvTransform * gh.cov * splitCovInvTransform;
	JacobiSVD<CovMatrix> decomposedCov(stretchedCov,ComputeFullU);
	const vector<GaussianHypothesis<1> > * independentHypotheses[dim];
	vector<GaussianHypothesis<1> > temporaryVectorsOnStack[dim];
	double scaleRatios[dim];
	for(int i=0; i<dim; i++) {
		double variance = decomposedCov.singularValues()(i);
		if(variance<=1) {
			scaleRatios[i] = 1;
			Matrix<double,1,1> newMean; newMean<<0;
			Matrix<double,1,1> newVar; newVar<<variance;
			temporaryVectorsOnStack[i].push_back(GaussianHypothesis<1>(newMean,newVar,1));
			independentHypotheses[i] = &temporaryVectorsOnStack[i];
		} else {
			TableEntry<1> entry = table.getUpperEntry(variance);
			scaleRatios[i] = sqrt(variance/entry.variance);
			independentHypotheses[i] = &entry.hypotheses;
		}
	}

	bool finished = false;
	vector<GaussianHypothesis<1> >::const_iterator iterators[dim];
	for(int i=0; i<dim; i++) iterators[i] = independentHypotheses[i]->begin();
	while(!finished) {
		GaussianHypothesis<dim> * newHypothesis = new GaussianHypothesis<dim>();
		DiagonalMatrix<double,dim> baseCov;
		MeanMatrix baseMean;
		newHypothesis->weight = 1;
		for(int i=0; i<dim; i++) {
			baseMean(i) = iterators[i]->mean(0)*scaleRatios[i];
			baseCov.diagonal()(i) = iterators[i]->cov(0,0)*scaleRatios[i]*scaleRatios[i];
			newHypothesis->weight *= iterators[i]->weight;
		}
		newHypothesis->weight*=gh.weight;
		newHypothesis->mean=splitCovTransform*(decomposedCov.matrixU()*baseMean)+gh.mean;
		CovMatrix newStretchedCov = decomposedCov.matrixU()*baseCov*decomposedCov.matrixU().transpose();
		newHypothesis->cov=splitCovTransform*newStretchedCov*splitCovTransform;
		result.push_front(*newHypothesis);
		for(int i=0; i<dim; i++) { // This loop is "incrementing" the "iterators" vector to the next "state" in an abstract sense
			iterators[i]++;
			if(iterators[i]==independentHypotheses[i]->end()) {
				iterators[i]=independentHypotheses[i]->begin();
				if(i==dim-1) finished = true;
			} else break;
		}
	}

}


#endif /* MULTIHYPDIST_HPP_ */
