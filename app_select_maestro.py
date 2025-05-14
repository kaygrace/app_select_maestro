#!/usr/bin/env python

import matplotlib
matplotlib.use('Agg')
from matplotlib.pyplot import *
import numpy as np

import glob

from itertools import chain, combinations

def app_get(filename):
    app = float(filename[7:])
    return app

def get_info(file0):
    f=open(file0,'r')
    line=f.readline()
    ncols = len(line.split())
    nstars = (ncols - 1)/2
    cols = range(1,ncols-1,2)
    f.close()
    vars = np.loadtxt(file0,usecols=cols)
    target = vars[:,0]
    ndata = len(target)
    return nstars,ndata,ncols,cols,vars


'''def powerset(seq):
    """
    Returns all the subsets of this set. This is a generator.
    """
    if len(seq) <= 1:
        yield seq
        yield []
    else:
        for item in powerset(seq[1:]):
            yield [seq[0]]+item
            yield item'''

def powerset(seq):
    s = list(seq)
    return chain.from_iterable(combinations(s, r) for r in range(1, len(s)+1))

def check_files(nstars0,ndata0,ncols0,nstars,ndata,ncols):
    if (nstars0 != nstars):
        print("Error: different number of stars found")
        exit()
    elif (ndata0 != ndata):
        print("Error: different data files found")
        exit()
    elif (ncols0 != ncols):
        print("Error: different number of columns found")
        exit()

def comb_mask(amask):
    mask0 = amask[0]
    for mask in amask:
        mask0 = mask | mask0
    mask0 = np.logical_not(mask0)
    ind = np.arange(len(mask0))
    ind_mask = ind[mask0]
    return ind_mask

def consecutive(ind_mask):
    i0 = np.arange(len(ind_mask)-1)
    ip = i0 + 1
    ind_cons = np.where(ind_mask[ip] == ind_mask[i0]+1)
    ind_cons2 = np.asarray(ind_cons)[0]
    return ind_cons2

def ratio(target_lc,comp_lc,ind_mask,ind_cons):
    ratio0 = target_lc[ind_mask]/comp_lc[ind_mask]
    normed = ratio0/np.mean(ratio0)
    diff = normed[ind_cons + 1] - normed[ind_cons]
    scat = np.sqrt(np.dot(diff,diff)/len(diff))
    return scat,normed

def lc_clip(lc,hi=2.0,low=0.6):
    med=np.median(lc)
    med2 = np.ma.masked_outside(lc,low*med,hi*med)
    return med2

def padlim(ax,aspect=1.618,pad=0.05):
    """
    Set limits on current axis object in a SM-like manner
    """
    lims = ax.dataLim
    xint = lims.intervalx
    yint = lims.intervaly
    deltax = xint[1]-xint[0]
    xpad = pad/aspect
    ypad = pad
    dx = xpad * deltax
    x0 = xint[0] - dx
    x1 = xint[1] + dx
    xlim(x0,x1)
    deltay = yint[1]-yint[0]
    dy = ypad * deltay
    y0 = yint[0] - dy
    y1 = yint[1] + dy
    ylim(y0,y1)

def comp_str(comp_list):
    cstr = ''
    for k,i in enumerate(comp_list):
        if k==0:
            cstr = '{0}'.format(i)
        else:
            cstr = '{0}+{1}'.format(cstr,i)
    return cstr

#pattern = 'a*pr[0-9]*.'
pattern = 'counts_[0-9]*'
files = glob.glob(pattern)

print('\nProcessing the following files:')
sfiles = '  '
for i,fi in enumerate(files):
    sfiles= sfiles + ' {0:11s}'.format(fi)
    if ((i+1) % 4) == 0:
        sfiles= sfiles + '\n  '

print(sfiles)

nfiles = len(files)


# Get number of stars, data points from first file
fi = files[0]
nstars0,ndata0,ncols0,cols0,vars0 = get_info(files[0])

nstars0 = int(nstars0)

# make list of all possible combinations of comparison stars
comps = np.arange(1,nstars0-1)
comp_combs = list(powerset(comps))

# delete empty set from the above list
del comp_combs[-1]

uebervars = np.zeros((nfiles,ndata0,nstars0))
apps = np.zeros((nfiles))

print('\nReading in data with the following apertures:')

# read in all the aperture data and store in uebervars
for i,fi in enumerate(files):
    nstars,ndata,ncols,cols,vars = get_info(fi)
    check_files(nstars0,ndata0,ncols0,nstars,ndata,ncols)
    uebervars[i,:,:] = vars[:,:]
    apps[i] = app_get(fi)

# find the lower index of all "good" light curve points for which the next light curve point is also "good"
comp_best = []
scat_best = []
np_arr = []
# Loop over all aperture sizes
for i,app in enumerate(apps):

    # loop over comparison stars
    amask = []
    for j in comps:
        comp_lc = uebervars[i,:,j] 
        #data = sigma_clip(comp_lc, sig=2, iters=None,cenfunc=np.median,varfunc=np.var)
        data = lc_clip(comp_lc)
        mask = np.ma.getmaskarray(data)
        amask.append(mask)

# find the indices that all of the "good" light curve points have in common
    ind_mask = comb_mask(amask)

# find the lower index of all "good" light curve points for which the next light curve point is also "good"
    ind_cons = consecutive(ind_mask)
    np_arr.append(len(ind_cons))

    scat_arr = []
    target_lc = uebervars[i,:,0] 
    comp0 = np.zeros((ndata0))
    # loop over comparison stars to produce "divided" light curves
    for comb_list in comp_combs:
        comp_lc = comp0
        ifail = 0
        for j in comb_list:
            new_lc = uebervars[i,:,j] 
            if np.any(new_lc == 0.0):
                ifail = 1
            comp_lc = comp_lc + new_lc

        if ifail == 1:
            #print 'comparison lc',comb_list,'has zeros'
            scat = 1.e99
        else:
            scat,norm_lc = ratio(target_lc,comp_lc,ind_mask,ind_cons)
        scat_arr.append(scat)

    scat_arr = np.array(scat_arr)
    jmin = np.argmin(scat_arr)
    comp_best.append(comp_combs[jmin])
    scat_best.append(scat_arr[jmin])


scat_best = np.array(scat_best)
np_arr = np.array(np_arr)

# replace nans with max values
scat_best[np.isnan(scat_best)] = np.nanmax(scat_best)

isort = np.argsort(apps)
app_size = apps[isort]
scatter = scat_best[isort]
np_vals = np_arr[isort]

print(app_size)

ibest = np.argmin(scat_best)
cstring = comp_str(comp_best[ibest])

print('\nOptimal aperture is {0} pixels, and optimal comparison star is {1}\n'.format(apps[ibest],cstring))

legstr = 'optimal comparison\n=' + cstring

fig=figure()
ax=fig.add_subplot(1,1,1)
plot(app_size,scatter,'o-')
padlim(ax)
xlabel('Aperture size (pixels)')
ylabel('Average scatter')
ax.legend([legstr],loc='best',frameon=False)
savefig('lc.pdf',bbox_inches='tight')
savefig('apertures.pdf',bbox_inches='tight')


# compute best (least scatter) divided light curve
target_lc = uebervars[ibest,:,0]
comp_lc = comp0
for j in comb_list:
    comp_lc = comp_lc + uebervars[ibest,:,j] 
tlc = lc_clip(target_lc)
tlc = tlc/np.mean(tlc)
clc = lc_clip(comp_lc)
clc = clc/np.mean(clc)
rat = tlc/clc
x = 10.*np.arange(len(rat))

fig=figure()
ax2=fig.add_subplot(1,1,1)
plot(x,rat,'-')
plot(x,tlc-0.2,'--')
plot(x,clc-0.4,':')
xlabel('time (sec)')
ylabel('Divided lightcurve')
ax2.legend(['divided lightcurve','raw target','raw comparison'],loc='best',frameon=False)
savefig('lc.pdf',bbox_inches='tight')
