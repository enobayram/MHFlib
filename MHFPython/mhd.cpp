/*
 * mhd.cpp
 *
 *  Created on: Jan 12, 2012
 *      Author: eba
 */

#include "mhfpython.h"
#include <MultiHypDist.hpp>
#include "GHListWrap.h"
#include <memory>
#include <iostream>


template<int dim>
struct MultiHypDistWrap {
	std::auto_ptr<MultiHypDist<dim> > mhd;
	MultiHypDist<dim> * mhdPtr;
	HypListWrap<dim> listWrap;
	MultiHypDistWrap(): mhd(new MultiHypDist<dim>()), mhdPtr(&(*mhd)), listWrap(mhd->GHlist) {}
	MultiHypDistWrap(MultiHypDist<dim> & mhd): mhdPtr(&mhd), listWrap(mhd.GHlist)  {}
	void split(const GaussianHypothesis<dim> & gh, HypListWrap<dim> & result, Matrix<double,dim,1> & splitCov, SplitTable<1> & table) {
		mhdPtr->split(gh,*(result.listPtr), splitCov,table);
	}
};

template <int dim>
void read_MultiHypDist(MultiHypDist<dim> & self, const char * filename) {
	std::ifstream the_file;
	the_file.open(filename);
	the_file>>self;
	the_file.close();
}

template <int dim>
void export_mhd() {
	string distname = "MultiHypDist" + lexical_cast<string>(dim);
	void (*split)(const GaussianHypothesis<dim> & gh, typename GaussianHypothesis<dim>::list & result, Matrix<double,dim,1> & splitCov, SplitTable<1> & table)
			= &MultiHypDist<dim>::split;
	void (MultiHypDist<dim>::*split_self) (Matrix<double,dim,1> & splitCov, SplitTable<1> & table) = &MultiHypDist<dim>::split;

    class_<MultiHypDist<dim>, boost::noncopyable >(distname.c_str())
		.def_readonly("GHlist", &MultiHypDist<dim>::GHlist)
   		.def("split", split_self)
   		.def("read", make_function(read_MultiHypDist<dim>))
   		.def("resample", &MultiHypDist<dim>::resample)
    ;

    def("split", split);
    def("uniformSplit", MultiHypDist<dim>::uniformSplit);

    export_hypothesis<dim>();
}

void export_mhds() {
	export_mhd<dim>();
}
