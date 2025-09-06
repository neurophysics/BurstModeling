# %%
"""
Data_preprocessing.py
Lukasz Radzinski
Charite Neurophysics Group, Berlin
Script for preprocessing single channel
MEG recordings
"""

# %%
import os
import meet
import scipy
import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
from scipy import signal as sig
import helper_scripts.helper_functions as helper_functions

# set global parameters for plots
plt.rcParams['figure.figsize'] = [12, 6]
plt.rcParams['savefig.dpi'] = 600

# %%
# select file to preprocess

subject = 'S6'
additional_plot_title = ''

data_input_folder = '../Data/raw_data'
data_output_folder = '../Data/cleaned_data'
plots_output_folder = '../Results/preprocessing'

plots_output_folder = os.path.join(plots_output_folder, subject)

srate = 20000
unit = 'B [fT]'
asd_unit = '[fT/√HZ]'

# %%
# load data
raw_data = helper_functions.readMEG(os.path.join(data_input_folder, subject+'_MEG_stim.dat'), s_rate=srate, num_chans=2)

# %%
offset_start = 10*srate
if(subject=='S5'):
    offset_start = 60*srate

# change polarity of the MEG data to
# make it consistent with the EEG data
if(subject=='S1' or
   subject=='S2' or
   subject=='S3' or
   subject=='S4'):
    XEG_raw_data = -raw_data[0][offset_start:]
else:
    XEG_raw_data = raw_data[0][offset_start:]
stimulus_data = raw_data[-1][offset_start:]

# %%
# add header to the plot function
def plt_header(main_title=''):

    title = 'Subject '+subject
    title += '\n'+main_title+additional_plot_title

    plt.title(title)

# %%
# show and save the plot function
def plt_show_save_fig(fig_name=None):

    if(fig_name):
        fig_name += '.png'
    else:
        plt_show_save_fig.counter += 1
        fig_name = 'Fig%02d.png' % plt_show_save_fig.counter

    print('--------------------\n'+fig_name)
    os.makedirs(plots_output_folder, exist_ok=True)
    plt.savefig(os.path.join(plots_output_folder, fig_name), bbox_inches='tight')
    plt.show()

plt_show_save_fig.counter = 0

# %%
# plot the whole XEG signal

data = XEG_raw_data
data_x = np.linspace(0, (len(data)-1)/srate, len(data))
data_y = data

plt_header('Raw MEG data')
plt.plot(data_x, data_y)
plt.xlabel('t [s]')
plt.ylabel(unit)
plt_show_save_fig()

# %%
# plot XEG and stimulus signals

data = XEG_raw_data[:srate]
data_x = np.linspace(0, (len(data)-1)/srate, len(data))
data_y = data

plt_header('Raw MEG data')
plt.plot(data_x, data_y)
plt.xlabel('t [s]')
plt.ylabel(unit)
plt_show_save_fig()


data = stimulus_data[:srate]
data_x = np.linspace(0, (len(data)-1)/srate, len(data))
data_y = data

plt_header('Stimulus channel')
plt.plot(data_x, data_y)
plt.xlabel('t [s]')
plt.ylabel('stimulus')
plt_show_save_fig()


data = XEG_raw_data[:srate]
data = data - np.mean(data)
data = data / np.std(data)
data_x = np.linspace(0, (len(data)-1)/srate, len(data))
data_y = data

plt_header('MEG and stimulus channel normalized')
plt.plot(data_x, data_y, label='MEG channel')
plt.xlabel('t [s]')
plt.ylabel('stimulus')

data = stimulus_data[:srate]
data = data - np.min(data)
data = data / np.max(data)
data *= 5
data_x = np.linspace(0, (len(data)-1)/srate, len(data))
data_y = data

plt.plot(data_x, data_y, label='stimulus channel')
plt.xlabel('t [s]')
plt.ylabel('normalized units')
plt_show_save_fig()

# %%
# get the stimuli positions
all_stimuli = ((stimulus_data[1:]>250000) & (stimulus_data[:-1]<250000)).nonzero()[0]

# get the marker, omit the first and last (avoid edge effects)
marker = all_stimuli[1:-1]

# %%
# remove the stimuli

interpolate_win_ms = [-2, 2]
interpolate_win = np.round(np.array(interpolate_win_ms) / 1000. * srate).astype(int)
XEG_stimuli_removed_data = meet.interpolateEEG(XEG_raw_data.copy(), all_stimuli[:1], interpolate_win)

interpolate_win_ms = [-3, 3]
interpolate_win = np.round(np.array(interpolate_win_ms) / 1000. * srate).astype(int)
XEG_stimuli_removed_data = meet.interpolateEEG(XEG_stimuli_removed_data, all_stimuli[1:], interpolate_win)

# %%
# plot signal without stimuli

data = XEG_raw_data
data_x = np.linspace(0, (len(data)-1)/srate, len(data))
data_y = data
plt.plot(data_x, data_y, label='with stimulation artifacts')

data = XEG_stimuli_removed_data
data_x = np.linspace(0, (len(data)-1)/srate, len(data))
data_y = data
plt.plot(data_x, data_y, label='without stimulation artifacts')

plt_header('Removing stimuli artifacts from MEG data')
plt.xlabel('t [s]')
plt.ylabel(unit)
plt.legend()
plt_show_save_fig()


data = XEG_raw_data[:srate]
data_x = np.linspace(0, (len(data)-1)/srate, len(data))
data_y = data
plt.plot(data_x, data_y, label='with stimulation artifacts')

data = XEG_stimuli_removed_data[:srate]
data_x = np.linspace(0, (len(data)-1)/srate, len(data))
data_y = data
plt.plot(data_x, data_y, label='without stimulation artifacts')

plt_header('Removing stimuli artifacts from MEG data')
plt.xlabel('t [s]')
plt.ylabel(unit)
plt.legend()
plt_show_save_fig()


start_time = marker[0]/srate - 0.01 
end_time = marker[0]/srate + 0.01

data = XEG_raw_data[int(start_time*srate):int(end_time*srate)]
data_x = np.linspace(start_time*1000, end_time*1000, len(data))
data_y = data
plt.plot(data_x, data_y, label='with stimulation artifacts')

data = XEG_stimuli_removed_data[int(start_time*srate):int(end_time*srate)]
data_x = np.linspace(start_time*1000, end_time*1000, len(data))
data_y = data
plt.plot(data_x, data_y, label='without stimulation artifacts')

plt_header('Removing stimuli artifacts from MEG data')
plt.xlabel('t [ms]')
plt.ylabel(unit)
plt.legend()
plt_show_save_fig()

# %%
# apply fc=1Hz hp filter to remove DC component

sos = sig.butter(2, 1, 'highpass', fs=srate, output='sos')
XEG_data = sig.sosfiltfilt(sos, XEG_stimuli_removed_data)

# %%
# plot signal without DC component

data = XEG_data
data_x = np.linspace(0, (len(data)-1)/srate, len(data))
data_y = data
plt.plot(data_x, data_y)

plt_header('MEG data fc=1Hz hp filtered')
plt.xlabel('t [s]')
plt.ylabel(unit)
plt_show_save_fig()

# %%
# calculate amplitude spectral density

def asd(data, nfft):

    yf, xf = mlab.psd(x=data, NFFT=nfft, Fs=srate, window=sig.windows.hann(nfft), noverlap=nfft//2)
    
    return xf, np.sqrt(yf)

# %%
# plot asd of signal after removing stimulus

data = XEG_data
nfft = 2**(int(np.log2(srate))-1)
xf, yf = asd(data, nfft)

plt_header('Cleaning MEG signal, stimulus artifact removed')
plt.plot(xf, yf)

plt.xscale('log')
plt.yscale('log')
plt.xlim((1,srate//2))
plt.ylim((0.1, 1000))
plt.xlabel('f [Hz]')
plt.ylabel('Amplitude Spectral Density %s' % asd_unit)
plt.grid(axis='x', which='major', linewidth=1.5)
plt.grid(axis='x', which='minor', linewidth=0.5)
plt.grid(axis='y', which='major', linewidth=1.5)
plt.grid(axis='y', which='minor', linewidth=0.5)
plt_show_save_fig()

# %%
# remove powerline artifacts / spectrum peaks

XEG_with_spectrum_peaks = XEG_data

## apply low-pass filter to remove potential artifact near Nyquist freqency
sos = sig.butter(4, 6000, 'lowpass', fs=srate, output='sos')
XEG_data = sig.sosfiltfilt(sos, XEG_data)
asd_peaks_combined = []

for i in range(3):
    # detect peaks in the spectrum
    nfft = 2**(int(np.log2(srate))-1)
    data = XEG_data
    xf, yf = asd(data, nfft)

    if(i > 0):
        yf_median = scipy.ndimage.median_filter(yf, size=100)
        yf_min_clipped = np.clip(yf, yf_median, None)

    asd_peaks = sig.find_peaks(yf, prominence=0.5)[0]
    asd_peaks = asd_peaks[xf[asd_peaks] > 80]
    asd_peaks_combined += asd_peaks.tolist()
    peaks_to_remove = xf[asd_peaks]


    print('Peaks to remove:', peaks_to_remove)

    if(len(peaks_to_remove) > 0):

        import mne
        info = mne.create_info(ch_names=['ch1'], sfreq=srate, ch_types=['eeg'])
        raw = mne.io.RawArray([XEG_data], info)

        # filtering
        raw = raw.notch_filter(freqs=peaks_to_remove, method="spectrum_fit")
        XEG_data = raw['ch1'][0][0]

# reinterpolate the stimulus to remove filtering artifacts
XEG_data = meet.interpolateEEG(XEG_data, marker, interpolate_win)

## apply low-pass filter to remove potential artifact near Nyquist freqency
sos = sig.butter(4, 5000, 'lowpass', fs=srate, output='sos')
XEG_data = sig.sosfiltfilt(sos, XEG_data)

print('Peaks removed:', asd_peaks_combined)

# %%
# plot asd of signal without powerline artifacts / spectrum peaks

data = XEG_with_spectrum_peaks
nfft = 2**(int(np.log2(srate))-1)
xf, yf = asd(data, nfft)

plt_header('Cleaning MEG signal, removing powerline artifacts / spectrum peaks')
plt.scatter(xf[asd_peaks_combined], yf[asd_peaks_combined], marker='x', color='red', label='peaks to remove')
plt.plot(xf, yf, label='with artifacts')

data = XEG_data
xf, yf = asd(data, nfft)

plt.plot(xf, yf, label='artifacts removed')
plt.xscale('log')
plt.yscale('log')
plt.xlim((1,srate//2))
plt.ylim((0.1, 1000))
plt.xlabel('f [Hz]')
plt.ylabel('Amplitude Spectral Density %s' % asd_unit)
plt.legend()
plt.grid(axis='x', which='major', linewidth=1.5)
plt.grid(axis='x', which='minor', linewidth=0.5)
plt.grid(axis='y', which='major', linewidth=1.5)
plt.grid(axis='y', which='minor', linewidth=0.5)
plt_show_save_fig()

# %%
# apply 450Hz-850Hz band-pass filter to extract
# high-frequency band (sigma band) and 
# high-frequency somatosensory evoked response (sigma burst)

lfreq_sigma = 450
rfreq_sigma = 850
sigma_freq_range_str = '%sHz-%sHz' % (lfreq_sigma, rfreq_sigma)

sigma_band_data = meet.iir.butterworth(XEG_data, fs=(lfreq_sigma-50, rfreq_sigma+50),
                                        fp=(lfreq_sigma, rfreq_sigma), s_rate=srate)

# %%
# extract trials to remove outliers

whole_trial_len = int(np.round(np.mean(np.diff(all_stimuli))))
whole_trial_win_samples = [0,whole_trial_len]
whole_trial_t = (np.arange(whole_trial_win_samples[0], whole_trial_win_samples[1], 1)/float(srate)*1000)

XEG_whole_trials = meet.epochEEG(XEG_data, marker, whole_trial_win_samples)
XEG_with_spectrum_peaks_whole_trials = meet.epochEEG(XEG_with_spectrum_peaks, marker, whole_trial_win_samples)
sigma_band_whole_trials = meet.epochEEG(sigma_band_data, marker, whole_trial_win_samples)

sigma_win_ms = [10, 35]
sigma_win_samples = np.round(np.array(sigma_win_ms)/1000.*srate).astype(int)
sigma_burst_trials = meet.epochEEG(sigma_band_data, marker, sigma_win_samples)

# %%
# calculate sigma band whole trials rms and percentiles to remove outliers
sigma_band_trials_rms = np.sqrt(np.mean(sigma_band_whole_trials**2, axis=0))

sigma_band_rms_q25 = scipy.stats.scoreatpercentile(sigma_band_trials_rms, 25)
sigma_band_rms_q50 = np.median(sigma_band_trials_rms)
sigma_band_rms_q75 = scipy.stats.scoreatpercentile(sigma_band_trials_rms, 75)
sigma_band_rms_iqr = sigma_band_rms_q75 - sigma_band_rms_q25

# set a high threshold to remove only very outliers
threshold_sigma_band_rms = sigma_band_rms_q75 + 3*sigma_band_rms_iqr
not_outliers_sigma_band = sigma_band_trials_rms <= threshold_sigma_band_rms

data_y = sigma_band_trials_rms
plt.plot(np.clip(data_y, 0, 60), linewidth=1)
plt.axhline(threshold_sigma_band_rms, linewidth=1, alpha=0.5, color='red', label='threshold')

plt_header('MEG sigma band whole trials rms, outliers rejection candidates, n=%d' % np.sum(data_y > threshold_sigma_band_rms))
plt.xlabel('trial number')
plt.ylabel(unit)
plt.legend()
plt_show_save_fig()

# %%
# final outliers estimation

not_outliers = not_outliers_sigma_band

not_outliers = np.convolve(not_outliers, np.ones(5)/5, mode='same')
not_outliers = (not_outliers > 0.999)

not_outliers_arr = []
temp_arr = []

last_value = False
for i in range(len(not_outliers)):
    if(not_outliers[i] == False and last_value == True):
        not_outliers_arr.append(temp_arr)
        temp_arr = []

    if(not_outliers[i] == True):
        temp_arr.append(i)
    
    last_value = not_outliers[i]

for i in range(len(not_outliers_arr)):
    if(len(not_outliers_arr[i]) < 20):
        not_outliers[not_outliers_arr[i]] = False

plt_header('Final outliers rejection, trials rejected = %d, trials remained = %d' % (np.sum(not_outliers == False), np.sum(not_outliers)))
plt.plot(not_outliers)
plt.xlabel('trial number')
plt.ylabel('not outlier')
plt_show_save_fig()

# %%
# reject outliers

XEG_whole_trials = XEG_whole_trials[:,not_outliers]
XEG_with_spectrum_peaks_whole_trials = XEG_with_spectrum_peaks_whole_trials[:,not_outliers]
sigma_band_whole_trials = sigma_band_whole_trials[:,not_outliers]
sigma_burst_trials = sigma_burst_trials[:,not_outliers]
sigma_band_trials_rms = sigma_band_trials_rms[not_outliers]
#sigma_burst_trials_rms = sigma_burst_trials_rms[not_outliers]

print(XEG_whole_trials.shape)
print(XEG_with_spectrum_peaks_whole_trials.shape)
print(sigma_band_whole_trials.shape)
print(sigma_burst_trials.shape)
print(sigma_band_trials_rms.shape)
#print(sigma_burst_trials_rms.shape)

data_y = sigma_band_trials_rms
plt.plot(np.clip(data_y, 0, 60), linewidth=1)
plt.axhline(threshold_sigma_band_rms, linewidth=1, alpha=0.5, color='red', label='threshold')

plt_header('MEG sigma band whole trials rms, after outliers rejection, n=%d' % np.sum(data_y > threshold_sigma_band_rms))
plt.xlabel('trial number')
plt.ylabel(unit)
plt.legend()
plt_show_save_fig()

# %%
# concatenate trials without outliers

XEG_data_no_outliers = np.concatenate(XEG_whole_trials.T)
XEG_with_spectrum_peaks_data_no_outliers = np.concatenate(XEG_with_spectrum_peaks_whole_trials.T)
all_stimuli_no_outliers = (np.arange(XEG_whole_trials.shape[1])*XEG_whole_trials.shape[0])
marker_no_outliers = all_stimuli_no_outliers[1:]
XEG_data_no_outliers = meet.interpolateEEG(XEG_data_no_outliers, marker_no_outliers, interpolate_win)
XEG_with_spectrum_peaks_data_no_outliers = meet.interpolateEEG(XEG_with_spectrum_peaks_data_no_outliers, marker_no_outliers, interpolate_win)

# %%
# plot asd of data with outliers removed

data = XEG_data
nfft = 2**(int(np.log2(srate))-1)
xf, yf = asd(data, nfft)

plt_header('Cleaning MEG signal, removing outliers')
plt.plot(xf, yf, label='with outliers')

data = XEG_data_no_outliers
xf, yf = asd(data, nfft)

plt.plot(xf, yf, label='outliers removed')
plt.xscale('log')
plt.yscale('log')
plt.xlim((1,srate//2))
plt.ylim((0.1, 1000))
plt.xlabel('f [Hz]')
plt.ylabel('Amplitude Spectral Density %s' % asd_unit)
plt.legend()
plt.grid(axis='x', which='major', linewidth=1.5)
plt.grid(axis='x', which='minor', linewidth=0.5)
plt.grid(axis='y', which='major', linewidth=1.5)
plt.grid(axis='y', which='minor', linewidth=0.5)
plt_show_save_fig()

# %%
# plot asd of data with outliers removed, compare data with and without spectrum peaks

data = XEG_with_spectrum_peaks_data_no_outliers
nfft = 2**(int(np.log2(srate))-1)
xf, yf = asd(data, nfft)

plt_header('Cleaning MEG signal, outliers removed, show removed spectrum peaks')
plt.plot(xf, yf, label='with spectrum peaks artifacts')

data = XEG_data_no_outliers
xf, yf = asd(data, nfft)

plt.plot(xf, yf, label='spectrum peaks removed')
plt.xscale('log')
plt.yscale('log')
plt.xlim((1,srate//2))
plt.ylim((0.1, 1000))
plt.xlabel('f [Hz]')
plt.ylabel('Amplitude Spectral Density %s' % asd_unit)
plt.legend()
plt.grid(axis='x', which='major', linewidth=1.5)
plt.grid(axis='x', which='minor', linewidth=0.5)
plt.grid(axis='y', which='major', linewidth=1.5)
plt.grid(axis='y', which='minor', linewidth=0.5)
plt_show_save_fig()

# %%
XEG_data = XEG_data_no_outliers
all_stimuli = all_stimuli_no_outliers
marker = marker_no_outliers

# %%
# create additional signal with evoked response removed

XEG_whole_trials_mean = np.mean(XEG_whole_trials, axis=-1)
XEG_data_no_mean = np.concatenate(XEG_whole_trials.T - XEG_whole_trials_mean)
XEG_data_no_mean = meet.interpolateEEG(XEG_data_no_mean, marker, interpolate_win)

data = XEG_data[:srate]
data_x = np.linspace(0, (len(data)-1)/srate, len(data))
data_y = data
plt.plot(data_x, data_y, label='Signal with evoked response')

data = XEG_data_no_mean[:srate]
data_x = np.linspace(0, (len(data)-1)/srate, len(data))
data_y = data
plt.plot(data_x, data_y, label='Signal with evoked response removed')

plt_header('Creating MEG signal with evoked response removed')
plt.xlabel('t [s]')
plt.ylabel(unit)
plt.legend()
plt_show_save_fig()

# %%
# plot broadband evoked response

data = XEG_whole_trials_mean
data_x = np.linspace(0, (len(data)-1)/srate, len(data))*1000
data_y = data
plt.plot(data_x, data_y)
plt_header('Average of broadband trials, n = %d' % len(XEG_whole_trials.T))
plt.xlabel('t [ms]')
plt.ylabel(unit)
plt.ylim((-600, 600))
plt.xlim((0, 300))
plt.grid()
plt_show_save_fig()

data = XEG_whole_trials_mean[:srate//10]
data_x = np.linspace(0, (len(data)-1)/srate, len(data))*1000
data_y = data
plt.plot(data_x, data_y)
plt_header('Average of broadband trials, n = %d' % len(XEG_whole_trials.T))
plt.xlabel('t [ms]')
plt.ylabel(unit)
plt.ylim((-600, 600))
plt.xlim((0, 100))
plt.grid()
plt_show_save_fig()

# %%
# plot sigma burst (high-frequency somatosensory evoked response)

sigma_band_whole_trials_mean = np.mean(sigma_band_whole_trials, axis=-1)

sigma_rms_st = np.sqrt(np.mean(sigma_band_whole_trials[srate*15//1000:srate*30//1000]**2, axis=0))
noise_rms_st = np.sqrt(np.mean(sigma_band_whole_trials[srate*50//1000:srate*100//1000]**2, axis=0))
snnr_st = np.mean(sigma_rms_st/noise_rms_st)

sigma_rms_er = np.sqrt(np.mean(sigma_band_whole_trials_mean[srate*15//1000:srate*30//1000]**2))
noise_rms_er = np.sqrt(np.mean(sigma_band_whole_trials_mean[srate*50//1000:srate*100//1000]**2))
snnr_er = sigma_rms_er/noise_rms_er

data = sigma_band_whole_trials_mean
data_x = np.linspace(0, (len(data)-1)/srate, len(data))*1000
data_y = data
plt.plot(data_x, data_y, label='sigma burst\nsnnr st: %6.3f\nsnnr er: %.3f' % (snnr_st, snnr_er))
plt_header('Sigma burst, average of trials, 450Hz-850Hz bp filter, n = %d' % len(sigma_band_whole_trials.T))
plt.xlabel('t [ms]')
plt.ylabel(unit)
plt.ylim((-30, 30))
plt.xlim((0, 300))
plt.grid()
plt.legend(prop={'family': 'DejaVu Sans Mono'})
plt_show_save_fig()

data = sigma_band_whole_trials_mean[:srate//10]
data_x = np.linspace(0, (len(data)-1)/srate, len(data))*1000
data_y = data
plt.plot(data_x, data_y, label='sigma burst\nsnnr st: %6.3f\nsnnr er: %.3f' % (snnr_st, snnr_er))
plt_header('Sigma burst, average of trials, 450Hz-850Hz bp filter, n = %d' % len(sigma_band_whole_trials.T))
plt.xlabel('t [ms]')
plt.ylabel(unit)
plt.ylim((-30, 30))
plt.xlim((0, 100))
plt.grid()
plt.legend(prop={'family': 'DejaVu Sans Mono'})
plt_show_save_fig()


data = sigma_band_whole_trials_mean[:srate//10]
data_x = np.linspace(0, (len(data)-1)/srate, len(data))*1000
data_y = data
plt.plot(data_x, data_y, label='sigma burst\nsnnr st: %6.3f\nsnnr er: %.3f' % (snnr_st, snnr_er))
plt_header('Sigma burst, average of trials, 450Hz-850Hz bp filter, n = %d' % len(sigma_band_whole_trials.T))
plt.xlabel('t [ms]')
plt.ylabel(unit)
plt.ylim((-30, 30))
plt.xlim((0, 50))
plt.grid()
plt.legend(prop={'family': 'DejaVu Sans Mono'})
plt_show_save_fig()

# %%
# plot sigma burst (high-frequency somatosensory evoked response)

data = sigma_band_whole_trials[:srate//10].std(-1)
data_x = np.linspace(0, (len(data)-1)/srate, len(data))*1000
data_y = data
plt.plot(data_x, data_y)
plt_header('Sigma burst, standard deviation of trials, 450Hz-850Hz bp filter, n = %d' % len(sigma_band_whole_trials.T))
plt.xlabel('t [ms]')
plt.ylabel(unit)
plt.ylim((0, 20))
plt.xlim((0, 100))
plt.grid()
plt_show_save_fig()


data = sigma_band_whole_trials[:srate//10].std(-1)
data_x = np.linspace(0, (len(data)-1)/srate, len(data))*1000
data_y = data
plt.plot(data_x, data_y)
plt_header('Sigma burst, standard deviation of trials, 450Hz-850Hz bp filter, n = %d' % len(sigma_band_whole_trials.T))
plt.xlabel('t [ms]')
plt.ylabel(unit)
plt.ylim((0, 20))
plt.xlim((0, 50))
plt.grid()
plt_show_save_fig()

# %%
# plot sigma burst single trials stack on each other

plt.rcParams['image.cmap'] = 'coolwarm'

plt_header('Sigma burst, all trials')
data = sigma_band_whole_trials[:srate//10,]
data_x = np.linspace(0, 1000*(len(data)-1)/srate, len(data))

limit = 30
plt.pcolormesh(data_x, np.arange(len(data.T)), data.T, vmin=-limit, vmax=limit)
plt.ylabel('Trial number')
plt.xlabel('Time [ms]')
plt.xlim(0,100)
clb = plt.colorbar(extend='both')
clb.set_label(unit)
plt_show_save_fig()

start_trial = 400
end_trial = 600
plt_header('Sigma burst, exemplary trials')
data = sigma_band_whole_trials[:srate//10,start_trial:end_trial+1]
data_x = np.linspace(0, 1000*(len(data)-1)/srate, len(data))

plt.pcolormesh(data_x, np.arange(start_trial, end_trial+1), data.T, vmin=-limit, vmax=limit)
plt.ylabel('Trial number')
plt.xlabel('Time [ms]')
plt.xlim(0,100)
clb = plt.colorbar(extend='both')
clb.set_label(unit)
plt_show_save_fig()

# %%
# save the preprocessed data

stimulus_new = np.zeros(len(XEG_data))
stimulus_new[marker_no_outliers] = 1.0

out_data = np.stack((XEG_data, XEG_data_no_mean, stimulus_new))
os.makedirs(data_output_folder, exist_ok=True)
np.save(os.path.join(data_output_folder, subject+'.npy'), out_data)


