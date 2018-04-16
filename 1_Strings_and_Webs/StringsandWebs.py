#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  2 19:58:22 2018

@author: Stella
"""
import matplotlib.patches as mpatches
import numpy as np
import matplotlib.pyplot as plt
import urllib
import urllib2
period = 1.7
url = 'http://nesssi.cacr.caltech.edu/cgi-bin/getcssconedbid_release2.cgi'
values= {'Name' : 'Her X-1', 'OUT' : 'web', 'SHORT' : 'short', 'DB':'photcat'}
data = urllib.urlencode(values)
req = urllib2.Request(url, data)
response = urllib2.urlopen(req)
the_page = response.read()
print the_page
parse1 = the_page.split('ID<th>Mag<th>Magerr<th>RA<th>Dec<th>MJD</tr>')
parse2 = parse1[1].split('\n')
del parse2[-3:] ##removes ['</table><br><p>', '<p><br><p></HTML>', '']
data = np.empty([len(parse2),6]) ## ID Mag Magerr RA Dec MJD
for i in range(len(parse2)):
    parse3 = parse2[i].lstrip('<tr><td>')
    parse3 = parse3.rstrip('</tr>')
    data[i] = parse3.split('<td>')
    
    plt.figure(0)
    plt.errorbar(data[i,5],data[i,1],data[i,2],fmt='ro')
    phase = data[i,5] % period
                
    plt.figure(1)
    plt.errorbar(phase,data[i,1],data[i,2],fmt='yo')
plt.figure(0)    
plt.xlabel('Date(MJD)')
plt.ylabel('V mag')
plt.gca().invert_yaxis()
plt.grid()
red_patch = mpatches.Patch(color='red', label='Her X-1')
plt.legend(handles= [red_patch])
plt.title('Magnitude versus time for Her X-1')
plt.savefig('Her_X-1.pdf')
plt.close()

plt.figure(1)
plt.xlabel('Phase')
plt.ylabel('V mag')
plt.gca().invert_yaxis()
plt.grid()
yellow_patch = mpatches.Patch(color='yellow', label=('Period = 1.7 days'))
plt.legend(handles= [yellow_patch])
plt.title('Light Curve for Her X-1')
plt.savefig('Periodplot.pdf')
plt.close()
from astropy.io.votable import parse, parse_single_table
votable = parse('votable.xml',pedantic=False)
table = parse_single_table('votable.xml')
times = table.array['ObsTime']
mag = table.array['Mag']
timeandmag = np.empty([len(times),2])
timeandmag[:,0] = times[:,0]
timeandmag[:,1]= mag[:,0]
print timeandmag