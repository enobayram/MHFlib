#include <boost/python.hpp>
#include <GaussianHypothesis.h>
#include <iostream>

void printGH(const GaussianHypothesis<3> & in) {
	std::cout<<in;
}

void printVoid() {std::cout<<"hello!\n";}

BOOST_PYTHON_MODULE(TmpModule)
{
	using namespace boost::python;
	//class_<GaussianHypothesis<3> >("GaussianHypothesis3");
	def("print_stuff",printVoid);
	def("print_stuff",printGH);

}
