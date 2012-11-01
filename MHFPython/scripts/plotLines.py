# -*- coding: utf-8 -*-
"""
Created on Fri Mar  9 14:22:28 2012

@author: eba
"""

def plotLines():
    class line_entry():
       def __init__(self,startx, starty, endx, endy):
            self.startx=startx;
            self.starty=starty;
            self.endx=endx;
            self.endy=endy;
       def __repr__(self):
           return "("+str(self.startx)+","+str(self.starty)+')-('+str(self.endx)+","+str(self.endy)+")"
    
    
    lines = [];
    import fileinput
    import re
    for line in fileinput.input('ulmsserver.ini'):
        fields = line.split()
        if (len(fields)>0 and fields[0] == 'addline'):
            startx = float(re.search('startx=(.*)',fields[1]).group(1))
            starty = float(re.search('starty=(.*)',fields[2]).group(1))
            endx = float(re.search('endx=(.*)',fields[3]).group(1))
            endy = float(re.search('endy=(.*)',fields[4]).group(1))
            lines.append(line_entry(startx,starty,endx,endy))
     
    import matplotlib.pyplot
    for line in lines:
        matplotlib.pyplot.plot([line.startx,line.endx],[line.starty,line.endy],'w',linewidth=3)