/*
 * SplitTable.h
 *
 *  Created on: Nov 29, 2011
 *      Author: eba
 */

#ifndef SPLITTABLE_H_
#define SPLITTABLE_H_

#include "GaussianHypothesis.h"
#include <map>
#include <vector>
#include <string>
#include <boost/foreach.hpp>
#include <iostream>
#include <fstream>
#include <cstdio>

inline std::string getDefaultTable() {
	#include "defaultTable"
	return std::string((char *) kl1e_2table,kl1e_2table_len);

}

template<int dim>
struct TableEntry{
	const double & variance;
	const std::vector<GaussianHypothesis<dim> > & hypotheses;
	inline TableEntry(const typename std::map<const double,std::vector<GaussianHypothesis<dim> > >::iterator it): variance(it->first), hypotheses(it->second){}
};

template<int dim>
class SplitTable {
public:
	std::map<const double, std::vector<GaussianHypothesis<dim> > > table;
	SplitTable(const char * filename) {
		if(strcmp(filename,"")==0) return;
		std::ifstream fs;
		fs.open(filename, std::ios::in);
		if(!fs) {
			std::cout<<"the file: "<<filename<<" is not found, so, split table is not available for multi-hypothesis purposes\n";
			return;
		}
		readTable(fs);
		fs.close();
	}

	SplitTable() {
		std::string tableString = getDefaultTable();
		std::stringstream ss(tableString, std::stringstream::in);
		readTable(ss);
	}

	void readTable(std::istream & fs) {
		while(!fs.eof()) {
			double variance;
			int count;
			fs>>variance>>count;
			if(fs.eof()) break;
//			std::cout<<" "<<variance<<" "<<count<<" ";
			std::vector<GaussianHypothesis<dim> > hypotheses(count);
			BOOST_FOREACH(GaussianHypothesis<dim> & hyp, hypotheses) {
				fs>>hyp;
			}
			table[variance] = hypotheses;
		}
	}

	TableEntry<dim> getUpperEntry(double variance) {
		return TableEntry<dim>(table.upper_bound(variance));
	}

	virtual ~SplitTable(){}
};



#endif /* SPLITTABLE_H_ */
