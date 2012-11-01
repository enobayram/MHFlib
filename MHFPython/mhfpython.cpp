/*
 * mhf_python.cpp
 *
 *  Created on: Dec 2, 2011
 *      Author: eba
 */

#include "mhfpython.h"
#include <boost/python.hpp>

void export_tables();
void export_GHLists();
void export_ukfs();
void export_matrices();
void export_mhds();
void export_visualization();
void export_transformations();

BOOST_PYTHON_MODULE(MHFPython)
{
	export_tables();
	export_GHLists();
	export_ukfs();
	export_matrices();
	export_mhds();
	export_visualization();
	export_transformations();
}
