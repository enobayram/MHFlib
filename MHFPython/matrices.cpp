/*
 * matrices.cpp
 *
 *  Created on: Jan 12, 2012
 *      Author: eba
 */
#include "mhfpython.h"
#include <eigen3/Eigen/Dense>
#include <GaussianHypothesis.h>
#include <boost/python/stl_iterator.hpp>
#include <list>
#include <numpy/noprefix.h>
#include <string>

using namespace Eigen;


typedef GaussianHypothesis<dim>::MeanMatrix MeanMatrix;
typedef GaussianHypothesis<dim>::CovMatrix CovMatrix;

typedef Matrix<double,1,1> ScalarMatrix;


template<int rows, int cols>
void matrix_assign(Matrix<double, rows, cols>& m, boost::python::object o) {
    // Turn a Python sequence into an STL input range
    boost::python::stl_input_iterator<double > begin(o), end;
    std::list<double> l; l.assign(begin,end);
    std::list<double>::iterator it = l.begin();
    for(int r = 0; r<rows; r++)
    	for(int c=0; c<cols; c++)
    		m(r,c) = *(it++);
}

template<int rows, int cols>
//boost::python::numeric::array *
PyObject *
pyArray(Matrix<double, rows, cols> & mat) {
	npy_intp N[] = {rows,cols};
	//return new array(handle<>(PyArray_SimpleNewFromData(2,N, PyArray_DOUBLE, mat.data())));
	//return static_cast<array>(handle<>(PyArray_SimpleNewFromData(2,N, PyArray_DOUBLE, mat.data())));
	return PyArray_SimpleNewFromData(2,N, PyArray_DOUBLE, mat.data()); // EIGEN_DEFAULT_TO_ROW_MAJOR is needed!!
}

template<int rows, int cols>
Matrix<double, rows,cols> * createMatrix(object o) {
	Matrix<double, rows, cols> * result = new Matrix<double, rows, cols>();
	for(int r = 0; r<rows; r++)
		for(int c = 0; c<cols; c++)
			(*result)(r,c) = extract<double>(o.attr("__getitem__")(r).attr("__getitem__")(c));
	return result;

}

template<int rows, int cols>
class export_proxy {
public:
	static bool exported;
};

template<int rows, int cols >
bool export_proxy<rows,cols>::exported = false;

template<int rows, int cols>
void export_matrix() {

	if(export_proxy<rows,cols>::exported) return;
	typedef Matrix<double, rows, cols> Mat;
	string name = "Matrix" + lexical_cast<string>(rows) + lexical_cast<string>(cols);

    double &(Mat::*MemberAccess) (typename Mat::Index) = &Mat::operator();
    class_<Mat>(name.c_str())
    	    .def("__init__", make_constructor(createMatrix<rows,cols>) )
			.def("__call__", MemberAccess, return_value_policy<copy_non_const_reference>())
    		.def(self_ns::str(self))
	;
    def("assign",matrix_assign<rows,cols>);
    def("pyArray", pyArray<rows,cols>, with_custodian_and_ward_postcall<0,1>());
    export_proxy<rows,cols>::exported=true;
}

template<int rows, int cols>
Matrix<double, rows, cols> * construct_matrix() {
	return new Matrix<double, rows, cols>();
}

template<int rows, int cols>
void export_matrix(string name) {
	def(name.c_str(), make_constructor(createMatrix<rows,cols>));
	def(name.c_str(), construct_matrix<rows, cols>, return_value_policy<manage_new_object >());
	export_matrix<rows, cols>();
}

void export_matrices() {

	import_array();
	array::set_module_and_type("numpy", "ndarray");

	export_matrix<1,1>("ScalarMatrix");
	export_matrix<dim,1>("MeanMatrix");
	export_matrix<dim,dim>("CovMatrix");
	export_matrix<inputdim,1>();
	export_matrix<procnoisedim,1>();
	export_matrix<procnoisedim,procnoisedim>();
	export_matrix<measdim,1>();
	export_matrix<measdim,measdim>();
	export_matrix<1,1>();
	export_matrix<2,1>();

}
