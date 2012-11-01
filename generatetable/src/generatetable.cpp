//============================================================================
// Name        : generatetable.cpp
// Author      : 
// Version     :
// Copyright   : Your copyright notice
// Description : Hello World in C++, Ansi-style
//============================================================================

#include <iostream>
#include <fstream>
#include <stdio.h>
#include <algorithm>
#include <math.h>
//#include <boost/lambda/bind.hpp>
//#include <boost/lambda/lambda.hpp>
#include <boost/spirit/home/phoenix.hpp>
#include <boost/function.hpp>
#include <boost/foreach.hpp>
#include <eigen3/Eigen/Dense>
#include <time.h>
#include <vector>
#include <map>
#include <cstring>
#include <signal.h>
#include <limits>
#include <boost/program_options.hpp>

//using namespace boost::lambda;
using namespace boost::phoenix;
using namespace Eigen;
namespace po = boost::program_options;
using std::vector;
using std::cout;
using std::endl;
using std::cerr;
using std::ostream;
using std::ofstream;
using std::fstream;
using std::map;
using std::max;
using std::numeric_limits;
using boost::result_of;

namespace bp = boost::phoenix;

//boost::lambda::placeholder1_type X;
actor<argument<0> > X;


bool quitting = false;

const double pi = 3.141592653589793238462643383279502884197169399;
void quit(int);

int numberOfThreads=3;

class Uniform {
public:
	double width;
	Uniform(double width): width(width){}
	inline double operator()(double x) const{
		return 1/width;
	}
	double getIntBegin() const {
		return -width/2;
	}
	double getIntEnd() const {
		return width/2;
	}
};

class Gaussian {
public:
	double mean;
	double var;
	double weight;
	Gaussian(double var): mean(0), var(var), weight(1){}
	Gaussian(double mean, double var): mean(mean),var(var), weight(1){}
	Gaussian(double mean, double var, double weight): mean(mean),var(var), weight(weight){}
	inline double operator()(double x) const{
		return 1/sqrt(2*pi*var)*exp(-(x-mean)*(x-mean)/(2*var));
	}
	void setMean(double mean) {this->mean=mean;}

	inline bool operator<(const Gaussian & other) const {
		return mean<other.mean;
	}
	double getIntBegin() const {
		return mean - 4 * sqrt(var); //integralBegin
	}
	double getIntEnd() const {
		return mean + 4 * sqrt(var); //integralEnd
	}
};

typedef map<double, vector<Gaussian> >::iterator tableIterator;
typedef vector<Gaussian>::iterator vectorIterator;

inline double normal(double x,double mean, double var) {
	return 1/sqrt(2*pi*var)*exp(-(x-mean)*(x-mean)/(2*var));
}

template <class F>
inline typename boost::result_of<F(double)>::type integrate(double from, double to, double increment, F const &f, typename boost::result_of<F(double)>::type accumulator = 0) {
	int N = (to-from)/increment;
	double x;
	int i;
	typename boost::result_of<F(double)>::type perThreadAccumulator;
	#pragma omp parallel private(x,perThreadAccumulator,i) num_threads(numberOfThreads)
	{
		perThreadAccumulator = accumulator;
		#pragma omp for schedule(dynamic, N/40)
		for(i=0;i<N; i++) {
			x=from+i*increment;
			perThreadAccumulator += f(x);
		}
		#pragma omp critical
		accumulator += perThreadAccumulator;
	}
	accumulator*=increment;
	return accumulator;
}

//template <class F>
//inline double integrate_(double from, double to, double increment, F const &f) {
//	double result=0;
//	int N = (to-from)/increment;
//	double x;
//	#pragma omp parallel for private(x) reduction(+:result) schedule(dynamic, N/40) num_threads(numberOfThreads)
//	for(int i=0;i<N; i++) {
//		x=from+i*increment;
//		result += f(x);
//	}
//	result*=increment;
//	return result;
//}


template <class Distribution>
VectorXd EMintegrand (double x, const Distribution& original, const vector<Gaussian>& splitted) {
	double denominator = 0;
	for (vector<Gaussian>::const_iterator it = splitted.begin(); it!=splitted.end(); ++it) {
		denominator+=(*it)(x)*it->weight;
	}
	if(denominator==0) return VectorXd::Zero(splitted.size());
	VectorXd result(splitted.size());
	const double constTerm = original(x)/denominator;
	for(uint i=0; i<splitted.size(); i++) {
		result(i) = splitted[i].weight*splitted[i](x)*constTerm;
	}
	return result;
}


//template <class Distribution>
//double EMintegrand (double x, int index, const Distribution& original, const vector<Gaussian>& splitted) {
//	double denominator = 0;
//	for (vector<Gaussian>::const_iterator it = splitted.begin(); it!=splitted.end(); ++it) {
//		denominator+=(*it)(x)*it->weight;
//	}
//	if(denominator==0) return 0;
//	return splitted[index].weight*splitted[index](x)*original(x)/denominator;
//}

template <class Distribution>
double DistIntegrand (double x, const Distribution& original, const vector<Gaussian>& splitted) {
	double originalValue = original(x);
	double approximatingValue = 0;
	BOOST_FOREACH(const Gaussian & hyp, splitted) {
		approximatingValue += hyp(x)*hyp.weight;
	}
	return originalValue*log(originalValue/approximatingValue);
}

double calculateChange(vector<Gaussian> & oldHyps, vector<Gaussian> & newHyps) {
	double change = 0;
	for(vectorIterator oldIt = oldHyps.begin(), newIt = newHyps.begin(); oldIt!= oldHyps.end(); oldIt++, newIt++) {
		change = max(change,fabs(oldIt->mean-newIt->mean));
		change = max(change,fabs(oldIt->var-newIt->var));
		change = max(change,fabs(oldIt->weight-newIt->weight));
	}
	return change;
}

void printHypotheses(vector<Gaussian> & hypotheses) {
	BOOST_FOREACH(Gaussian & g, hypotheses) {
		cout<< g.mean << ", "<<g.var<<": "<<g.weight<< " ||\n";
	}
}

template<class Distribution>
double EM(vector<Gaussian> & splitted, const Distribution original, double maxVar = 1, bool print = true) {
	cout<<"---- Starting a new EM ---- \n";
	double Dist = numeric_limits<double>::quiet_NaN(); // Bhattacharyya coefficient
	double ib = original.getIntBegin(); //integralBegin
	double ie = original.getIntEnd(); //integralEnd
	double iinc = (ie - ib) / 2000; //integralIncrement
	vector<Gaussian> oldHypotheses = splitted;

	for(int j=0; j<100000; j++) {
		if(print) cout<<"it "<<j<< ": ";

		VectorXd newWeights_ = integrate(ib,ie,iinc,bp::bind(EMintegrand<Distribution>,X,original,splitted),VectorXd::Zero(splitted.size()));
		VectorXd newMeans_ =   integrate(ib,ie,iinc,bp::bind(EMintegrand<Distribution>,X,original,splitted)*X,VectorXd::Zero(splitted.size())).cwiseQuotient(newWeights_);
		VectorXd secondMoments = integrate(ib,ie,iinc,bp::bind(EMintegrand<Distribution>,X,original,splitted)*X*X,VectorXd::Zero(splitted.size())).cwiseQuotient(newWeights_);
		VectorXd newVars_ = secondMoments - newMeans_.cwiseProduct(newMeans_);

		for (uint i=0; i<splitted.size(); ++i) {
			splitted[i].weight = newWeights_(i);
			splitted[i].mean = newMeans_(i);
			if(newVars_(i)<maxVar) splitted[i].var = newVars_(i);
			else {splitted[i].var=maxVar;}
		}
		double newDist = integrate(ib,ie,iinc,bp::bind(DistIntegrand<Distribution>,X,original,splitted));
		cout << newDist;
		double change = calculateChange(splitted,oldHypotheses);
		//if((Dist - newDist) < 1e-6) break;
		Dist = newDist;
		if(change<1e-5) break; else oldHypotheses = splitted;
		if(print) cout << "\r";
		fflush(stdout);
		if(quitting) {
			cout<<"\ninterrupted";
			break;
		}
	}
	cout<<"\n";
	printHypotheses(splitted);
	return Dist;
}

void expandHypotheses(vector<Gaussian> &hypotheses) {
	if(hypotheses.size()%2) { //odd case
		int middleIndex = hypotheses.size()/2;
		Gaussian &g = hypotheses.at(middleIndex);
		Gaussian g1(g.mean-sqrt(g.var), g.var/sqrt(2),g.weight/2);
		Gaussian g2(g.mean+sqrt(g.var), g.var/sqrt(2),g.weight/2);
		hypotheses.erase(hypotheses.begin()+middleIndex);
		hypotheses.push_back(g1);
		hypotheses.push_back(g2);
	} else { // even case
		int rightIndex = hypotheses.size()/2;
		int leftIndex = rightIndex-1;
		Gaussian &oldLeft = hypotheses.at(leftIndex);
		Gaussian &oldRight = hypotheses.at(rightIndex);
		Gaussian gm((oldLeft.mean+oldRight.mean)/2, (oldLeft.var+oldRight.var)/2, (oldLeft.weight+oldRight.weight)/4);
		Gaussian gl((oldLeft.mean-sqrt(oldLeft.var)),oldLeft.var/sqrt(2),oldLeft.weight/2);
		Gaussian gr((oldRight.mean+sqrt(oldRight.var)),oldRight.var/sqrt(2),oldRight.weight/2);
		hypotheses.erase(hypotheses.begin()+leftIndex);
		hypotheses.erase(hypotheses.begin()+rightIndex);
		hypotheses.push_back(gl);
		hypotheses.push_back(gm);
		hypotheses.push_back(gr);
	}
	std::sort(hypotheses.begin(), hypotheses.end());
}

void stretchHypotheses(vector<Gaussian>&hypotheses, double ratio) {
	BOOST_FOREACH(Gaussian & g, hypotheses) {
		g.mean*=sqrt(ratio);
	}
}

double linearIncrement(int step, double maxvariance, int tableSize) {return 1+(maxvariance-1)*(step+1)/(tableSize);}
double geometricIncrement(int step, double maxvariance, int tableSize) {
	double logmax = log(maxvariance);
	double logstep = logmax*(step+1)/tableSize;
	return exp(logstep);
}

bool nonSaturatedCriterion(vector<Gaussian> hypotheses) {
	BOOST_FOREACH(Gaussian & g, hypotheses) {
		if(g.var<1) return true;
	}
	return false;
}

bool KLdivUpperBoundCriterion(double KLdiv, double KLdivUpperBound) {
	return KLdiv < KLdivUpperBound;
}

enum Criterion { KLDIVUPPERBOUND, SATURATION};

template <class Distribution>
void fillTable(map<double,vector<Gaussian> > &table,int tableSize,bool geometricTableSteps, double maxvariance, double maxVar, Criterion criterion, double KLdivUpperBound) {
	vector <Gaussian> splitted;
	splitted.push_back(Gaussian(0,1,1));
	double oldVariance = 1;
	for(int i=0; i<tableSize && !quitting; i++) {
		double variance;
		if(geometricTableSteps) variance = geometricIncrement(i,maxvariance,tableSize);
		else variance = linearIncrement(i,maxvariance,tableSize);
		cout<<"\n\nGenerating Table Entry #"<< i+1 << " With Variance: "<<variance<< "\n";
		Distribution original(variance);
		stretchHypotheses(splitted,variance/oldVariance);

		bool isApproximatedWell = false;
		while(!isApproximatedWell && !quitting) {
			double KLdiv = EM(splitted, original, maxVar);
			switch (criterion) {
			case SATURATION: isApproximatedWell = nonSaturatedCriterion(splitted); break;
			case KLDIVUPPERBOUND:	isApproximatedWell = KLdivUpperBoundCriterion(KLdiv,KLdivUpperBound); break;
			default: std::cerr<<"Undefined criterion!!!"; exit(1);
			}
			if(!isApproximatedWell) expandHypotheses(splitted);
		}

		table[variance] = splitted;
		oldVariance=variance;
	}
}

int main(int argc, char *argv[]) {
	signal(SIGINT,quit);
	double maxVar=1;
	//	int iterations = 1000;
	//	int hypotheses = 5;
	double maxvariance = 2;
	int tableSize = 2;
	bool geometricTableSteps = true;
	bool uniformTable = false;
	double KLdivUpperBound = 1e-3;
	Criterion criterion = KLDIVUPPERBOUND;
	std::string filename="";
	
	// Declare the supported options.
	po::options_description desc("Allowed options");
	desc.add_options()
	    ("help", "produce help message")
	    ("output-file,o", po::value<std::string>(&filename), "The output file name, dumps the table to the standard output if none provided")
	    ("maxvariance,m", po::value<double>(&maxvariance)->default_value(2), "The maximum source Gaussian variance to be included in the table")
	    ("tablesize,s",   po::value<int>(&tableSize)->default_value(2), "The number of entries in the table")
	    ("geometric,g", "Indicates that the variance values for consequtive entries should increase geometrically (Default)")
	    ("linear,l", "Indicates that the variance values for consequtive entries should increase linearly")
	    ("numofthreads,n", po::value<int>(&numberOfThreads)->default_value(3), "The number of threads to use for the computation of the table")
	    ("usekldiv", "Use an upper-bound for the Kulbeck-Leibler distance of the mixture to stop refining a table entry(Default)")
	    ("usesaturation", "Use a special saturation condition to stop refining a table entry, the condition is that the resulting mixture starts to contain hypoteses that are narrower than the maximum allowed")
	    ("klupperbound,k", po::value<double>(&KLdivUpperBound)->default_value(1e-3), "The upper bound for the Kulbeck-Leibler distance of the splitted mixture to the original")
	    ("uniform,u", "Indicates that the original distribution to be splitted is a uniform distribution")
	    ("width,w", po::value<double>(&maxvariance)->default_value(2), "The width of the uniform distribution (The distribution is 1/width from/to (-/+) 1/(2*width)")
	;

	po::variables_map vm;
	po::store(po::parse_command_line(argc, argv, desc), vm);
	po::notify(vm);    

	if (vm.count("help")) {
	    cout << desc << "\n";
	    return 1;
	}

	if (vm.count("linear")) {
		geometricTableSteps = false;
	} 
	
	if (vm.count("usesaturation")) {
		criterion = SATURATION;
	} 

	map<double,vector<Gaussian> > table;

	if(uniformTable) {
		cout<<"creating a uniform table\n";
		fillTable<Uniform>(table,tableSize,geometricTableSteps,maxvariance,maxVar,criterion,KLdivUpperBound);
	} else	{
		fillTable<Gaussian>(table,tableSize,geometricTableSteps,maxvariance,maxVar,criterion,KLdivUpperBound);
	}

	//	vector <Gaussian> splitted;
	//	splitted.push_back(Gaussian(0,1,1));
	//	double oldVariance = 1;
	//	for(int i=0; i<tableSize && !quitting; i++) {
	//		double variance;
	//		if(geometricTableSteps) variance = geometricIncrement(i,maxvariance,tableSize);
	//		else variance = linearIncrement(i,maxvariance,tableSize);
	//		cout<<"\n\nGenerating Table Entry #"<< i+1 << " With Variance: "<<variance<< "\n";
	//		Gaussian original(0,variance,1);
	//		stretchHypotheses(splitted,variance/oldVariance);
	//
	//		bool isApproximatedWell = false;
	//		while(!isApproximatedWell) {
	//			double KLdiv = EM(splitted, original, maxVar);
	//			switch (criterion) {
	//			case SATURATION: isApproximatedWell = nonSaturatedCriterion(splitted); break;
	//			case KLDIVUPPERBOUND:	isApproximatedWell = KLdivUpperBoundCriterion(KLdiv,KLdivUpperBound); break;
	//			default: std::cerr<<"Undefined criterion!!!"; exit(1);
	//			}
	//			if(!isApproximatedWell) expandHypotheses(splitted);
	//		}
	//
	//		table[variance] = splitted;
	//		oldVariance=variance;
	//	}

	cout<< endl;
	ofstream the_file;
	ostream* hypothesisout;
	if(filename=="")	hypothesisout=&cout;
	else {
		the_file.open(filename);
		hypothesisout=&the_file;
	}

	for(tableIterator it=table.begin(); it!=table.end(); it++) {
		*hypothesisout<<"\n"<<it->first<<" "<<it->second.size()<<"\n";
		BOOST_FOREACH(Gaussian & g, it->second) {
			*hypothesisout<<"\t"<<g.mean<<"  "<<g.var<< "  "<<g.weight<<"\n";
		}
	}

	//	//*hypothesisout << original.mean<< " " << original.var << " " <<original.weight << endl;
	//	for (uint i=0; i<splitted.size(); ++i) {
	//		*hypothesisout<< splitted[i].mean<< " " << splitted[i].var << " " << splitted[i].weight << endl;
	//	}
	if(the_file.is_open()) the_file.close();

	return 0;
}

void quit(int in) {
	quitting=true;
}
