import scipy, pylab, os

def stft(x, fs, framesz, hop):
    framesamp = int(framesz*fs)
    hopsamp = int(hop*fs)
    w = scipy.hamming(framesamp)
    X = scipy.array([scipy.fft(w*x[i:i+framesamp])
                     for i in range(0, len(x)-framesamp, hopsamp)])
    return X

import matplotlib.pyplot as plt
from scipy.io import wavfile # get the api
from scipy.fftpack import fft
from pylab import *
import csv

def f(filename, theClass=1):
    fs, data = wavfile.read(filename) # load the data
    # b=[(ele/2**8.)*2-1 for ele in data] # this is 8-bit track, b is now normalized on [-1,1)
    print "Sample rates is: "
    print fs
    X = stft(data, fs, 256.0/fs, 256.0/fs)
    X = X[:,0:(X.shape[1]/2)]
    shortTimeFFT = scipy.absolute(X.T)
    shortTimeFFT = scipy.log10(shortTimeFFT)

    # Plot the magnitude spectrogram.
    pylab.figure()
    pylab.imshow(shortTimeFFT, origin='lower', aspect='auto',
                 interpolation='nearest')
    pylab.xlabel('Time')
    pylab.ylabel('Frequency')
    savefig(filename+'SFFT.png',bbox_inches='tight')

    features = mean(shortTimeFFT, axis=1)
    pylab.figure()
    pylab.plot(features,'r')
    savefig(filename+'AFFT.png',bbox_inches='tight')

    with open(filename+'.csv', 'w') as fp:
        a = csv.writer(fp, delimiter=',')
        row = pylab.transpose(features)
        row = pylab.append(row, theClass)
        a.writerow(row)

myFolder = "/Volumes/source/Python/speaker/dataP3/"
theClass = 3

import glob
files = glob.glob(myFolder + '*.wav')
for ele in files:
    f(ele, theClass)
os.chdir(myFolder)
os.system('cat *.csv > all.csv')
quit()

# Then call: ( cat dataP1/all.csv; echo; cat dataP2/all.csv; echo; cat dataP3/all.csv ) > both.csv