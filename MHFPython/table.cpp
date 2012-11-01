/*
 * table.cpp
 *
 *  Created on: Jan 12, 2012
 *      Author: eba
 */

#include "mhfpython.h"
#include <boost/python.hpp>
#include <boost/range.hpp>
#include <SplitTable.h>
#include <vector>

using std::vector;
using namespace boost::python;
using namespace boost::python::numeric;

double get_variance(const TableEntry<1>& self) {return self.variance;}
vector<GaussianHypothesis<1> >get_hypotheses(const TableEntry<1>& self) {return self.hypotheses;}

void export_tables()
{
    class_<SplitTable<1> >("SplitTable1",init<const char *>())
    		.def(init<>())
    		.def("getUpperEntry", &SplitTable<1>::getUpperEntry)
    ;
    class_<TableEntry<1> >("TableEntry1",no_init)
    	    .add_property("variance", &get_variance)
    	    .add_property("hypotheses", &get_hypotheses)
    ;
    class_<vector<GaussianHypothesis<1> > >("GHvector")
    		.def("__iter__", iterator<vector<GaussianHypothesis<1> > >())
    		.def("__len__", &vector<GaussianHypothesis<1> >::size)
    ;
    export_hypothesis<1>();
}
