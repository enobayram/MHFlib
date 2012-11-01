/*
 * GHList.cpp
 *
 *  Created on: Jan 12, 2012
 *      Author: eba
 */

#include "mhfpython.h"
#include <boost/python.hpp>
#include <boost/range.hpp>
#include "GHListWrap.h"

typedef GaussianHypothesis<dim>::list HypList;

void mergeLists(HypList & self, HypList & other){
	self.splice_after(self.begin(), other);
}

GaussianHypothesis<dim> & push_gaussian(HypList & self, const GaussianHypothesis<dim> & gh) {
	GaussianHypothesis<dim> * newGaussian = new GaussianHypothesis<dim>(gh);
	self.push_front(*newGaussian);
	return * newGaussian;
}

template<int dim>
void export_GHList() {
	string name = "GH" + lexical_cast<string>(dim) + "list";
    class_<HypList, boost::noncopyable>(name.c_str())
    		.def("__iter__", iterator<HypList,return_value_policy<reference_existing_object> >())
    		.def("__len__", &HypList::size)
    		.def("push_front", make_function(push_gaussian, return_value_policy<reference_existing_object>()))
    		.def("push_front", make_function(mergeLists))
    ;
}

void export_GHLists() {
	export_GHList<dim>();
}
