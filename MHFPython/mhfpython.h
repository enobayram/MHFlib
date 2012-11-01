/*
 * mhfpython.h
 *
 *  Created on: Jan 12, 2012
 *      Author: eba
 */

#ifndef MHFPYTHON_H_
#define MHFPYTHON_H_

#define EIGEN_DEFAULT_TO_ROW_MAJOR
#define EIGEN_DONT_ALIGN_STATICALLY

#include <boost/python.hpp>
#include <boost/python/numeric.hpp>
#include <boost/lexical_cast.hpp>
#include <GaussianHypothesis.h>


using namespace boost::python;
using namespace boost::python::numeric;
using std::string;
using boost::lexical_cast;


const int dim = 3, procnoisedim = 2, inputdim = 2, measdim = 1;

template <int dim>
void export_hypothesis() {
    string hypname = "GaussianHypothesis" + lexical_cast<string>(dim);
    class_<GaussianHypothesis<dim> >(hypname.c_str())
    		.def_readonly("mean", &GaussianHypothesis<dim>::mean)
    		.def_readonly("cov",&GaussianHypothesis<dim>::cov)
    		.def_readwrite("weight", &GaussianHypothesis<dim>::weight)
    		.def(self_ns::str(self))
    ;
}

#endif /* MHFPYTHON_H_ */
