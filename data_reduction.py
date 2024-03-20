# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt#
import pandas as pd
from scipy.signal import find_peaks

BackgroundTable = pd.read_csv('data/edips_1p4nm_DoC_ref_240228_000.dpt', sep=',', header=None)

def calc_absorption(data, offset=0.3):
    SampleTable = pd.read_csv(data, sep=',', header=None)
    #BackgroundTable = pd.read_csv('data/edips_1p4nm_DoC_ref_240228_000.dpt', sep=',', header=None)

    transmission = SampleTable[1]/BackgroundTable[1] + offset
    absorption = np.log10(1/transmission)
    return absorption

sampleA_NO = calc_absorption('data/edips_1p4nm_A_NO_240228_000.dpt')
sampleA_O = calc_absorption('data/edips_1p4nm_A_O_240228_000.dpt')
sampleA_precentrif = calc_absorption('data/edips_1p4nm_A_ref_240228_000.dpt') #, offset=0.32

sampleD_NO = calc_absorption('data/edips_1p4nm_D_NO_240228_000.dpt')
sampleD_O = calc_absorption('data/edips_1p4nm_D_O_240228_000.dpt')
sampleD_precentrif = calc_absorption('data/edips_1p4nm_D_ref_240228_000.dpt') #, offset=0.311

sample_1128 = calc_absorption('data/edips_1p4nm_centrif1128_Max_240228_000.dpt') #, offset=0.32
sample_1207 = calc_absorption('data/edips_1p4nm_centrif1207_Max_240228_000.dpt')

background_ev =  BackgroundTable[0]/8065.54
background_cm = BackgroundTable[0]


def finding_peaks(spectrum, w=40):
    peak_max_dict = find_peaks(spectrum, width=w)
    peaks = list(peak_max_dict[0])
    return peaks

peaks_A_NO = finding_peaks(sampleA_NO)
peaks_A_O = finding_peaks(sampleA_O)
peaksA_precentrif = finding_peaks(sampleA_precentrif)

peaks_D_NO = finding_peaks(sampleD_NO)
peaks_D_O = finding_peaks(sampleD_O)
peaksD_precentrif = finding_peaks(sampleD_precentrif)


fig = plt.figure(figsize=(15,10))
ax = fig.add_subplot(111)
plt.scatter(background_ev, sampleA_NO, s=50, label='w/o OptiPrep')
plt.scatter(background_ev, sampleA_O, s=50, label='w/ OptiPrep')
plt.scatter(background_ev, sampleA_precentrif, s=50, label='precentrif.')
plt.scatter(background_ev[peaks_A_O], sampleA_O[peaks_A_O], c='black', s=70, label='peaks')
plt.scatter(background_ev[peaks_A_NO], sampleA_NO[peaks_A_NO], c='black', s=70)
plt.scatter(background_ev[peaksA_precentrif], sampleA_precentrif[peaksA_precentrif], c='black', s=70)
plt.xlabel('energy / eV', fontsize=25)
plt.ylabel('absorbance / AU', fontsize=25)
plt.xlim(8000/8065.54, 23750/8065.54)
plt.ylim(0.42, 0.525)
plt.gca().invert_xaxis()
plt.title('Batch 2', fontsize=25)
plt.legend(loc=3, fontsize=25)
plt.show()

fig = plt.figure(figsize=(15,10))
ax = fig.add_subplot(111)
plt.scatter(background_ev, sampleD_NO, s=50, label='w/o OptiPrep')
plt.scatter(background_ev, sampleD_O, s=50, label='w/ OptiPrep')
plt.scatter(background_ev, sampleD_precentrif, s=50, label='precentrif.')
plt.scatter(background_ev[peaks_D_O], sampleD_O[peaks_D_O], c='black', s=70, label='peaks')
plt.scatter(background_ev[peaks_D_NO], sampleD_NO[peaks_D_NO], c='black', s=70)
plt.scatter(background_ev[peaksD_precentrif], sampleD_precentrif[peaksD_precentrif], c='black', s=70)
plt.xlabel('energy / eV', fontsize=25)
plt.ylabel('absorbance / AU', fontsize=25)
plt.xlim(8000/8065.54, 23750/8065.54)
plt.ylim(0.47, 0.525)
plt.gca().invert_xaxis()
plt.title('Batch 1', fontsize=25)
plt.legend(loc=3, fontsize=25)
plt.show()

fig = plt.figure(figsize=(15,10))
ax = fig.add_subplot(111)
plt.scatter(background_ev, sample_1128, s=1, label='1128')
plt.scatter(background_ev, sample_1207, s=1, label='1207')
plt.xlabel('wavenumber', fontsize=25)
plt.ylabel('absorbance', fontsize=25)
plt.xlim(8000/8065.54, 16000/8065.54)
plt.ylim(0.40, 0.495)
plt.gca().invert_xaxis()
plt.title('Others (no precentrif. measured)', fontsize=25)
plt.legend(loc=3, fontsize=25)
plt.show()

fig = plt.figure(figsize=(15,10))
ax = fig.add_subplot(111)
plt.scatter(background_ev, sampleA_NO/sampleA_precentrif, s=1, label='A_NO/REF')
plt.scatter(background_ev, sampleA_O/sampleA_precentrif, s=1, label='A_O/REF')
#plt.scatter(background_ev, sampleA_O/sampleA_NO, s=1, label='A_O/A_NO')
#plt.scatter(background_ev, sampleA_precentrif, s=1, label='A_precentrif')
plt.xlabel('wavenumber', fontsize=25)
plt.ylabel('absorbance', fontsize=25)
plt.xlim(8000/8065.54, 16000/8065.54)
plt.ylim(0.88, 0.955)
plt.gca().invert_xaxis()
plt.title('Sample A/REF', fontsize=25)
plt.legend(loc=3, fontsize=25)
plt.show()

fig = plt.figure(figsize=(15,10))
ax = fig.add_subplot(111)
#plt.scatter(background_ev, sampleA_NO/sampleA_precentrif, s=1, label='A_NO/REF')
#plt.scatter(background_ev, sampleA_O/sampleA_precentrif, s=1, label='A_O/REF')
plt.scatter(background_ev, sampleD_O/sampleD_NO, s=50, label='Batch 1')
plt.scatter(background_ev, sampleA_O/sampleA_NO, s=50, label='Batch 2')
plt.xlabel('energy / eV', fontsize=25)
plt.ylabel('absorbance / AU', fontsize=25)
plt.xlim(8000/8065.54, 16000/8065.54)
plt.ylim(0.998, 1.007)
plt.gca().invert_xaxis()
plt.title('(w/ O)/(w/o O)', fontsize=25)
plt.legend(loc=3, fontsize=25)
plt.show()


fig = plt.figure(figsize=(15,10))
ax = fig.add_subplot(111)
plt.scatter(background_ev, sampleA_NO-sampleA_precentrif, s=1, label='A_NO-REF')
plt.scatter(background_ev, sampleA_O-sampleA_precentrif, s=1, label='A_O-REF')
#plt.scatter(background_ev, sampleA_precentrif, s=1, label='A_precentrif')
plt.xlabel('wavenumber', fontsize=25)
plt.ylabel('absorbance', fontsize=25)
plt.xlim(8000/8065.54, 16000/8065.54)
plt.ylim(-0.05, -0.02)
plt.gca().invert_xaxis()
plt.title('Sample A', fontsize=25)
plt.legend(loc=3, fontsize=25)
plt.show()

fig = plt.figure(figsize=(15,10))
ax = fig.add_subplot(111)
plt.scatter(background_ev, sampleA_NO/sampleA_precentrif, s=50, label='(2 w/o O)/REF')
plt.scatter(background_ev, sampleA_O/sampleA_precentrif, s=50, label='(2 w/ O)/REF')
plt.scatter(background_ev, sampleD_NO/sampleD_precentrif, s=50, label='(1 w/o O)/REF')
plt.scatter(background_ev, sampleD_O/sampleD_precentrif, s=50, label='(1 w/ O)/REF')
#plt.scatter(background_ev, sampleA_O/sampleA_NO, s=1, label='A_O/A_NO')
#plt.scatter(background_ev, sampleA_precentrif, s=1, label='A_precentrif')
plt.xlabel('energy / eV', fontsize=25)
plt.ylabel('absorbance / AU', fontsize=25)
plt.xlim(8000/8065.54, 16000/8065.54)
plt.ylim(0.90, 0.975)
plt.gca().invert_xaxis()
plt.title('Sample/REF', fontsize=25)
plt.legend(loc=3, fontsize=25)
plt.show()

fig = plt.figure(figsize=(15,10))
ax = fig.add_subplot(111)
plt.scatter(background_ev, (sampleA_NO/sampleA_precentrif)/(sampleD_NO/sampleD_precentrif), s=50, label='w/o O')
plt.scatter(background_ev, (sampleA_O/sampleA_precentrif)/(sampleD_O/sampleD_precentrif), s=50, label='w/ O')
#plt.scatter(background_ev, sampleD_NO/sampleD_precentrif, s=50, label='D_NO/REF')
#plt.scatter(background_ev, sampleD_O/sampleD_precentrif, s=50, label='D_O/REF')
#plt.scatter(background_ev, sampleA_O/sampleA_NO, s=1, label='A_O/A_NO')
#plt.scatter(background_ev, sampleA_precentrif, s=1, label='A_precentrif')
plt.xlabel('energy / eV', fontsize=25)
plt.ylabel('absorbance / AU', fontsize=25)
plt.xlim(8000/8065.54, 16000/8065.54)
plt.ylim(0.972, 0.987)
plt.gca().invert_xaxis()
plt.title('Batch2 / Batch 1', fontsize=25)
plt.legend(loc=3, fontsize=25)
plt.show()

df = pd.DataFrame({'eV': background_ev,
                   'Absorbance': sampleD_O})
df.to_csv('sampleD_O.csv', index=False) 













