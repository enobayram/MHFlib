/*
 * transformations.cpp
 *
 *  Created on: Jan 18, 2012
 *      Author: eba
 */

#include "mhfpython.h"

#include "transformations.h"

void export_transformations() {
	class_<odometryTransformation, bases<transformations<3,2,2> > >("odometryTransformation",init<double,double,double>());
	class_<beaconMeasurement, bases<transformations<3,1,1> > >("beaconMeasurement",init<const Vector2d &>());
}
