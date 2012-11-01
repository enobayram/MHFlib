/*
 * GHListWrap.h
 *
 *  Created on: Jan 12, 2012
 *      Author: eba
 */

#ifndef GHLISTWRAP_H_
#define GHLISTWRAP_H_

#include <GaussianHypothesis.h>
#include <memory>
#include <list>


template< int dim>
struct HypListWrap {
	typedef typename GaussianHypothesis<dim>::list HypList;
	std::list<boost::python::handle<> > handles;
	std::auto_ptr<HypList> list;
	HypList * listPtr;
	HypListWrap(): list(new HypList()) {listPtr = &(*list);}
	HypListWrap(HypList & list) {
		listPtr = & list;
	}
	void push_front(boost::python::api::object & o) {
		extract<GaussianHypothesis<dim> &> gh(o);
		if(gh.check()) {
			listPtr->push_front(gh);
			handles.push_front(boost::python::handle<>(borrowed(o.ptr())));
			return;
		}
		extract<HypListWrap<dim> &> l(o);
		if(l.check()) {
			HypListWrap<dim> & otherList = l;
			listPtr->splice_after(listPtr->begin(), *(otherList.listPtr));
			handles.splice(handles.begin(), otherList.handles);
			return;
		}
		((GaussianHypothesis<dim> &) (gh)).weight; // Generate error if control reaches here
	}
	typedef typename HypList::iterator iterator;
	iterator begin() {return listPtr->begin();}
	iterator end() {return listPtr->end();}
	int size() {return listPtr->size();}
};

#endif /* GHLISTWRAP_H_ */
