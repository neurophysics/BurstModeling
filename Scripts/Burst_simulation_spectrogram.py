# %%
"""
Burst_simulation_spectrogram.py
Lukasz Radzinski
Charité Neurophysics Group, Berlin
Script for analysis of burst
time-frequency resolved data
"""

# %%
import os
import sys
import meet
import scipy
import numpy as np
import pandas as pd
from tqdm import trange
import matplotlib.pyplot as plt
from scipy import signal as sig
from BurstModel import BurstModel
import matplotlib.ticker as plticker
import helper_scripts.helper_functions as helper_functions


# set global parameters for plots

plt.rcParams['figure.figsize'] = [12, 6]
plt.rcParams['savefig.dpi'] = 600

# %%
# general settings

# select subject [S1-S5]
subject = 'S1'

additional_file_info = ''
additional_title_info = ''

data_input_folder = '../Data/cleaned_data'
plots_output_folder = '../Results/burst_simulation_spectrogram'

additional_plot_title = ''

srate = 20000

meg_unit = 'B [fT]'
meg_asd_unit = '[fT/√Hz]'

# %%
# add header to the plot
def plt_header(main_title='', use_suptitle = False, fontsize=12):

    title = main_title
    
    if(use_suptitle):
        plt.suptitle(title, fontsize=fontsize)
    else:
        plt.title(title, fontsize=fontsize)

# %%
# show and save the plot
def plt_show_save_fig(fig_name=None):

    if(fig_name):
        fig_name += '.png'
    else:
        plt_show_save_fig.counter += 1
        fig_name = 'fig%02d.png' % plt_show_save_fig.counter

    print('--------------------\n'+fig_name)
    os.makedirs(plots_output_folder, exist_ok=True)
    plt.savefig(os.path.join(plots_output_folder, fig_name), bbox_inches='tight')
    plt.show()

plt_show_save_fig.counter = 0

# %%
# load MEG and marker data

meg_data_file = os.path.join(data_input_folder, subject+'.npy')
meg_stim_data = np.load(meg_data_file)

meg_data = meg_stim_data[0]
marker = meet.getMarker(meg_stim_data[-1])

# %%
burst_prop_df = pd.read_csv('burst_subjects.csv')

lfreq_sigma = burst_prop_df[burst_prop_df['Subject']==subject]['Freq_start'].values[0]
rfreq_sigma = burst_prop_df[burst_prop_df['Subject']==subject]['Freq_end'].values[0]

sigma_freq_range = [lfreq_sigma, rfreq_sigma]

burst_time_start = burst_prop_df[burst_prop_df['Subject']==subject]['Time_start'].values[0]
burst_time_end = burst_prop_df[burst_prop_df['Subject']==subject]['Time_end'].values[0]

burst_win_ms = [burst_time_start, burst_time_end]

# %%
meg_data_hilb = sig.hilbert(meg_data)

# FIR filter
# apply band-pass filter to extract
# high-frequency band (sigma band)
# extend the band ±100Hz to obtain
# wider burst band shown on the TF plot
sigma_fir_coeffs = sig.firwin(303, [lfreq_sigma-100, rfreq_sigma+100], pass_zero=False, fs=srate)
meg_sigma_data = sig.filtfilt(sigma_fir_coeffs, 1.0, meg_data_hilb)

# %%
medium_win_ms = [0,100]
medium_win_samples = np.round(np.array(medium_win_ms)/1000.*srate).astype(int)
sigma_medium_trials = meet.epochEEG(meg_sigma_data, marker, medium_win_samples)
data_t_medium_ms = np.linspace(medium_win_ms[0], medium_win_ms[1], sigma_medium_trials.shape[-2]+1)[:-1]

burst_win_samples = np.round(np.array(burst_win_ms)/1000.*srate).astype(int)
sigma_burst_trials = meet.epochEEG(meg_sigma_data, marker, burst_win_samples)
data_t_burst_ms = np.linspace(burst_win_ms[0], burst_win_ms[1], sigma_burst_trials.shape[-2]+1)[:-1]

# %%
# calculate indices in medium trials that correspond to the sigma burst time window

burst_mt_smpl = [np.nonzero(data_t_medium_ms == burst_win_ms[0])[0][0],
              np.nonzero(data_t_medium_ms == burst_win_ms[1])[0][0]]
burst_mt_smpl = np.array(burst_mt_smpl)

# set simulated burst offset position in relation the physiological burst
sigma_sim_offset_ms = 30
burst_sim_mt_smpl = np.array(burst_mt_smpl)+sigma_sim_offset_ms*srate//1000
data_t_burst_sim_ms = data_t_burst_ms + sigma_sim_offset_ms

# %%
model_variables_subj = {}

model_variables_subj['S1'] = [1.394, 0.322, 0.000, 0.000, 0.519, 2.777]
model_variables_subj['S2'] = [2.163, 0.326, 0.000, 0.000, 0.252, 1.396]
model_variables_subj['S3'] = [0.640, 0.385, 0.000, 0.000, 0.366, 2.031]
model_variables_subj['S4'] = [0.818, 0.354, 0.000, 0.000, 0.483, 2.140]
model_variables_subj['S5'] = [0.510, 0.402, 0.000, 0.000, 0.667, 2.354]

model_variables = model_variables_subj[subject]
[div_steep, div_pos, ecavs, eclvs, lcavs, lclvs] = model_variables
div_pos_ms = div_pos*(burst_win_ms[1]-burst_win_ms[0])+burst_win_ms[0]

# %%
bm = BurstModel(data_t_medium_ms, sigma_medium_trials, burst_win_ms, sigma_sim_offset_ms, srate, seed=60)

[loss, division_curve, later_comp_sigma_er, lcabs, earlier_comp_sigma_er, ecabs, sigma_bursts_sim,
sigma_sim_medium_trials] = [i.numpy() for i in bm.calculate_model_output(model_variables)]

model_out_static = bm.calculate_model_output([10,0,0,0,0,0])
sigma_stat_medium_trials = model_out_static[-1].numpy()
sigma_bursts_stat = model_out_static[-2].numpy()
evoked_response = model_out_static[2].numpy()

# %%
# select time windows in ms for trials, stimulus time = 0
st_full_win_ms = [-100,300]
st_targ_win_ms = [0,100]
st_ref_win_ms = [100,200]

st_full_win_samples = np.round(np.array(st_full_win_ms)/1000.*srate).astype(int)
st_targ_win_samples = np.round(np.array(st_targ_win_ms)/1000.*srate).astype(int)
st_ref_win_samples = np.round(np.array(st_ref_win_ms)/1000.*srate).astype(int)

brb_st_full_trials = meet.epochEEG(meg_data_hilb, marker, st_full_win_samples)
brb_st_targ_trials = meet.epochEEG(meg_data_hilb, marker, st_targ_win_samples)
brb_st_ref_trials = meet.epochEEG(meg_data_hilb, marker, st_ref_win_samples)

data_t_st_full_ms = np.linspace(st_full_win_ms[0], st_full_win_ms[1], brb_st_full_trials.shape[-2]+1)[:-1]
data_t_st_targ_ms = np.linspace(st_targ_win_ms[0], st_targ_win_ms[1], brb_st_targ_trials.shape[-2]+1)[:-1]
data_t_st_ref_ms = np.linspace(st_ref_win_ms[0], st_ref_win_ms[1], brb_st_ref_trials.shape[-2]+1)[:-1]

# %%
burst_st_smpl = [np.nonzero(data_t_st_full_ms == burst_win_ms[0])[0][0],
                    np.nonzero(data_t_st_full_ms == burst_win_ms[1])[0][0]]
burst_st_smpl = np.array(burst_st_smpl)

burst_sim_st_smpl = burst_st_smpl+sigma_sim_offset_ms*srate//1000

burst_stat_st_smpl = burst_st_smpl+60*srate//1000

# %%
brb_sim_st_full_trials = brb_st_full_trials.copy()

brb_sim_st_full_trials[burst_sim_st_smpl[0]:burst_sim_st_smpl[1]] = (
brb_sim_st_full_trials[burst_sim_st_smpl[0]:burst_sim_st_smpl[1]] + sigma_bursts_sim)


brb_sim_st_full_trials[burst_stat_st_smpl[0]:burst_stat_st_smpl[1]] = (
brb_sim_st_full_trials[burst_stat_st_smpl[0]:burst_stat_st_smpl[1]] + sigma_bursts_stat)

brb_st_full_trials = brb_sim_st_full_trials

# %%
# create a custom sampling scheme for the S transform
def custom_sampling_meg(N):
    S_frange = [200, 2000]
    S_fnum = 50
    S_Nperperiod = 8
    wanted_freqs = np.exp(np.linspace(np.log(S_frange[0]),
        np.log(S_frange[1]), S_fnum))
    fftfreqs = np.fft.fftfreq(N, d=1./srate)
    # find the nearest frequency indices
    y = np.unique([np.argmin((w - fftfreqs)**2)
        for w in wanted_freqs])
    x = ((S_Nperperiod*fftfreqs[y]*N/float(srate))//2).astype(int)
    return x,y

# %%
l_limit_targ = np.argwhere(data_t_st_full_ms == st_targ_win_ms[0])[0][0]
r_limit_targ = np.argwhere(data_t_st_full_ms == st_targ_win_ms[1])[0][0]

l_limit_ref = np.argwhere(data_t_st_full_ms == st_ref_win_ms[0])[0][0]
r_limit_ref = np.argwhere(data_t_st_full_ms == st_ref_win_ms[1])[0][0]

coords, tf = meet.tf.gft(brb_st_full_trials, axis=0, sampling=custom_sampling_meg)

sampling_x, sampling_y = custom_sampling_meg(len(brb_st_full_trials))
fftfreqs = np.fft.fftfreq(len(brb_st_full_trials), d=1./srate)
used_frequencies = fftfreqs[sampling_y]

coords_targ_resampled = []
tf_targ_resampled = []

for i in range(len(coords[1])):
    if(coords[1][i] >= l_limit_targ and coords[1][i] <= r_limit_targ):
        coords_targ_resampled.append(coords[:,i])
        tf_targ_resampled.append(tf[:,i])

tf_ref_resampled = []
coords_ref_resampled = []

for i in range(len(coords[1])):
    if(coords[1][i] >= l_limit_ref and coords[1][i] <= r_limit_ref):
        coords_ref_resampled.append(coords[:,i])
        tf_ref_resampled.append(tf[:,i])

print(used_frequencies)

# %%
wanted_frequencies = np.exp(np.linspace(np.log(200), np.log(2000), 200))
wanted_frequencies = np.concatenate((wanted_frequencies, used_frequencies))
wanted_frequencies = np.unique(np.round(wanted_frequencies, 6))
print(wanted_frequencies)

# %%
tf_targ_resampled = np.stack(tf_targ_resampled).T
coords_targ_resampled = np.stack(coords_targ_resampled).T

coords_targ_resampled_rescaled = coords_targ_resampled.copy()

coords_targ_resampled_rescaled[0] = coords_targ_resampled_rescaled[0]*len(data_t_st_targ_ms)/len(data_t_st_full_ms)
coords_targ_resampled_rescaled[1] = coords_targ_resampled_rescaled[1] - l_limit_targ

tf_ref_resampled = np.stack(tf_ref_resampled).T
coords_ref_resampled = np.stack(coords_ref_resampled).T

coords_ref_resampled_rescaled = coords_ref_resampled.copy()

coords_ref_resampled_rescaled[0] = coords_ref_resampled_rescaled[0]*len(data_t_st_ref_ms)/len(data_t_st_full_ms)
coords_ref_resampled_rescaled[1] = coords_ref_resampled_rescaled[1] - l_limit_ref

# %%
coords_freqs_targ = []
coords_time_targ = []
coords_time_targ_org = []
tf_processed_targ = []

coords_freqs_targ_temp = []
coords_time_targ_temp = []
tf_processed_targ_temp = []

for i in range(len(coords_targ_resampled[0])):

    coords_freqs_targ_temp.append(coords_targ_resampled[0][i])
    coords_time_targ_temp.append(coords_targ_resampled[1][i])
    tf_processed_targ_temp.append(tf_targ_resampled[:,i])

    if(i == len(coords_targ_resampled[0])-1 or coords_targ_resampled[0][i] != coords_targ_resampled[0][i+1]):

        coords_freqs_targ.append(fftfreqs[np.array(coords_freqs_targ_temp).astype(int)])
        coords_time_targ.append(np.array(coords_time_targ_temp) - l_limit_targ)
        coords_time_targ_org.append(np.array(coords_time_targ_temp))
        tf_processed_targ.append(np.array(tf_processed_targ_temp).T)

        coords_freqs_targ_temp = []
        coords_time_targ_temp = []
        tf_processed_targ_temp = []


coords_freqs_ref = []
coords_time_ref = []
coords_time_ref_org = []
tf_processed_ref = []

coords_freqs_ref_temp = []
coords_time_ref_temp = []
tf_processed_ref_temp = []

for i in range(len(coords_ref_resampled[0])):

    coords_freqs_ref_temp.append(coords_ref_resampled[0][i])
    coords_time_ref_temp.append(coords_ref_resampled[1][i])
    tf_processed_ref_temp.append(tf_ref_resampled[:,i])

    if(i == len(coords_ref_resampled[0])-1 or coords_ref_resampled[0][i] != coords_ref_resampled[0][i+1]):

        coords_freqs_ref.append(fftfreqs[np.array(coords_freqs_ref_temp).astype(int)])
        coords_time_ref.append(np.array(coords_time_ref_temp) - l_limit_ref)
        coords_time_ref_org.append(np.array(coords_time_ref_temp))
        tf_processed_ref.append(np.array(tf_processed_ref_temp).T)

        coords_freqs_ref_temp = []
        coords_time_ref_temp = []
        tf_processed_ref_temp = []

# %%
# scale tf properly
for i in range(len(coords_time_targ)):
    tf_processed_targ[i] /= np.diff(coords_time_targ[i])[0]

# reconstruct waves
tf_waves_targ = tf_processed_targ.copy()
for i in range(len(tf_waves_targ)):
    tf_waves_targ[i] = tf_waves_targ[i]*np.exp(2j*np.pi*fftfreqs[sampling_y][i]*coords_time_targ_org[i]/srate)


# scale tf properly
for i in range(len(coords_time_ref)):
    tf_processed_ref[i] /= np.diff(coords_time_ref[i])[0]

# reconstruct waves
tf_waves_ref = tf_processed_ref.copy()
for i in range(len(tf_waves_ref)):
    tf_waves_ref[i] = tf_waves_ref[i]*np.exp(2j*np.pi*fftfreqs[sampling_y][i]*coords_time_ref_org[i]/srate)

# %%
from scipy import interpolate

tf_waves_targ_interp = []

for i in range(len(tf_waves_targ)):

    cs = interpolate.CubicSpline(coords_time_targ[i], tf_waves_targ[i], axis=-1)
    interpolated_sig_targ = cs(np.arange(0,len(data_t_st_targ_ms)))
    tf_waves_targ_interp.append(interpolated_sig_targ)
tf_waves_targ_interp = np.stack(tf_waves_targ_interp).swapaxes(0,1)


tf_waves_ref_interp = []

for i in range(len(tf_waves_ref)):

    cs = interpolate.CubicSpline(coords_time_ref[i], tf_waves_ref[i], axis=-1)
    interpolated_sig_ref = cs(np.arange(0,len(data_t_st_ref_ms)))
    tf_waves_ref_interp.append(interpolated_sig_ref)
tf_waves_ref_interp = np.stack(tf_waves_ref_interp).swapaxes(0,1)

# %%
env_avg_out = (np.mean(tf_waves_targ_interp, axis=0).T.__abs__()/
            np.mean(np.mean(tf_waves_ref_interp, axis=0).__abs__(), axis=-1)).T
interpolator = interpolate.interp1d(used_frequencies, env_avg_out, kind='linear', axis=0, fill_value='extrapolate')
env_avg_out_interp = interpolator(wanted_frequencies)


avg_env_out = (np.mean(tf_waves_targ_interp.__abs__(), axis=0).T/
            np.mean(np.mean(tf_waves_ref_interp.__abs__(), axis=0), axis=-1)).T
interpolator = interpolate.interp1d(used_frequencies, avg_env_out, kind='linear', axis=0, fill_value='extrapolate')
avg_env_out_interp = interpolator(wanted_frequencies)


std_env_out = (np.std(tf_waves_targ_interp.__abs__(), axis=0).T/
            np.mean(np.std(tf_waves_ref_interp.__abs__(), axis=0), axis=-1)).T
interpolator = interpolate.interp1d(used_frequencies, std_env_out, kind='linear', axis=0, fill_value='extrapolate')
std_env_out_interp = interpolator(wanted_frequencies)


std_out = (np.std(tf_waves_targ_interp.real, axis=0).T/
        np.mean(np.std(tf_waves_ref_interp.real, axis=0), axis=-1)).T
interpolator = interpolate.interp1d(used_frequencies, std_out, kind='linear', axis=0, fill_value='extrapolate')
std_out_interp = interpolator(wanted_frequencies)

# %%
def jackknife_avg(x):
    Nid, Ntr = x.shape
    means = np.abs((np.sum(x, 1)[:,np.newaxis] - x)/(Ntr - 1))
    return means

def jackknife_variance(x):
    Nid, Ntr = x.shape
    means = (np.sum(x, 1)[:,np.newaxis] - x)/(Ntr - 1)
    return (np.sum(x**2, 1)[:,np.newaxis] - x**2)/(Ntr-1) - means**2

def avg_test_stat(coords, sig, ref):
    # Careful, we need to give trial axis as 2nd axis
    assert sig.shape == ref.shape
    N = sig.shape[1]
    sig_jackknife_avg = jackknife_avg(sig)
    ref_jackknife_avg = jackknife_avg(ref)
    avg_diff = helper_functions.difference_Stransform(
        coords, sig_jackknife_avg, ref_jackknife_avg).T
    stat = avg_diff.mean(0)/np.sqrt(avg_diff.var(0) * (N-1))
    return stat

def var_test_stat(coords, sig, ref):
    # Careful, we need to give trial axis as 2nd axis
    assert sig.shape == ref.shape
    N = sig.shape[1]
    sig_jackknife_var = jackknife_variance(sig)
    ref_jackknife_var = jackknife_variance(ref)
    var_diff = helper_functions.difference_Stransform(
        coords, sig_jackknife_var, ref_jackknife_var).T
    stat = var_diff.mean(0)/np.sqrt(var_diff.var(0) * (N-1))
    return stat

# %%
env_avg_stat = avg_test_stat(
        coords_targ_resampled_rescaled,
        tf_targ_resampled.T,
        tf_ref_resampled.T)

avg_env_stat = avg_test_stat(
        coords_targ_resampled_rescaled,
        np.abs(tf_targ_resampled).T,
        np.abs(tf_ref_resampled).T)

std_env_stat = var_test_stat(
        coords_targ_resampled_rescaled,
        np.abs(tf_targ_resampled).T,
        np.abs(tf_ref_resampled).T)

std_stat = var_test_stat(
        coords_targ_resampled_rescaled,
        tf_targ_resampled.real.T,
        tf_ref_resampled.real.T)

# %%
all_tf = np.vstack([
    tf_targ_resampled,
    tf_ref_resampled
    ])

all_std_env_tf = np.vstack([
    np.abs(tf_targ_resampled) - np.abs(tf_targ_resampled).mean(0),
    np.abs(tf_ref_resampled) - np.abs(tf_ref_resampled).mean(0)
    ])

all_std_tf = np.vstack([
    tf_targ_resampled.real - tf_targ_resampled.real.mean(0),
    tf_ref_resampled.real - tf_ref_resampled.real.mean(0)
    ])

env_avg_boot_stat = []
avg_env_boot_stat = []
std_env_boot_stat = []
std_boot_stat = []

for _ in trange(200):
    idx1 = np.random.choice(all_tf.shape[0], size=tf_targ_resampled.shape[0],
            replace=True)
    idx2 = np.random.choice(all_tf.shape[0], size=tf_ref_resampled.shape[0],
            replace=True)

    env_avg_boot_stat.append(
            avg_test_stat(
                coords_targ_resampled_rescaled,
                all_tf[idx1].T,
                all_tf[idx2].T))
    avg_env_boot_stat.append(
            avg_test_stat(
                coords_targ_resampled_rescaled,
                np.abs(all_tf[idx1]).T,
                np.abs(all_tf[idx2].T)))
    std_env_boot_stat.append(
            var_test_stat(
                coords_targ_resampled_rescaled,
                all_std_env_tf[idx1].T,
                all_std_env_tf[idx2].T))
    std_boot_stat.append(
            var_test_stat(
                coords_targ_resampled_rescaled,
                all_std_tf[idx1].T,
                all_std_tf[idx2].T))

env_avg_boot_stat = np.array(env_avg_boot_stat)
avg_env_boot_stat = np.array(avg_env_boot_stat)
std_env_boot_stat = np.array(std_env_boot_stat)
std_boot_stat = np.array(std_boot_stat)

# %%
env_avg_p = helper_functions.stepdown_p(env_avg_stat, env_avg_boot_stat)

env_avg_p_val_arr = []
cnt = 0
for i in range(len(tf_waves_targ)):
    temp_arr = []
    for j in range(tf_waves_targ[i].shape[-1]):
        temp_arr.append(env_avg_p[cnt])
        cnt += 1
    env_avg_p_val_arr.append(temp_arr)

env_avg_p_val_interp = []
for i in range(len(tf_waves_targ)):

    interpolator = interpolate.interp1d(coords_time_targ[i], env_avg_p_val_arr[i], kind='nearest', fill_value='extrapolate')
    interpolated_sig = interpolator(np.arange(0,len(data_t_st_targ_ms)))
    env_avg_p_val_interp.append(interpolated_sig)

env_avg_p_val_interp = np.stack(env_avg_p_val_interp)
interpolator = interpolate.interp1d(used_frequencies, env_avg_p_val_interp, axis=0, kind='nearest', fill_value='extrapolate')
env_avg_p_val_interp = interpolator(wanted_frequencies)


avg_env_p = helper_functions.stepdown_p(avg_env_stat, avg_env_boot_stat)

avg_env_p_val_arr = []
cnt = 0
for i in range(len(tf_waves_targ)):
    temp_arr = []
    for j in range(tf_waves_targ[i].shape[-1]):
        temp_arr.append(avg_env_p[cnt])
        cnt += 1
    avg_env_p_val_arr.append(temp_arr)

avg_env_p_val_interp = []
for i in range(len(tf_waves_targ)):

    interpolator = interpolate.interp1d(coords_time_targ[i], avg_env_p_val_arr[i], kind='nearest', fill_value='extrapolate')
    interpolated_sig = interpolator(np.arange(0,len(data_t_st_targ_ms)))
    avg_env_p_val_interp.append(interpolated_sig)

avg_env_p_val_interp = np.stack(avg_env_p_val_interp)
interpolator = interpolate.interp1d(used_frequencies, avg_env_p_val_interp, axis=0, kind='nearest', fill_value='extrapolate')
avg_env_p_val_interp = interpolator(wanted_frequencies)


std_env_p = helper_functions.stepdown_p(std_env_stat, std_env_boot_stat)

std_env_p_val_arr = []
cnt = 0
for i in range(len(tf_waves_targ)):
    temp_arr = []
    for j in range(tf_waves_targ[i].shape[-1]):
        temp_arr.append(std_env_p[cnt])
        cnt += 1
    std_env_p_val_arr.append(temp_arr)

std_env_p_val_interp = []
for i in range(len(tf_waves_targ)):

    interpolator = interpolate.interp1d(coords_time_targ[i], std_env_p_val_arr[i], kind='nearest', fill_value='extrapolate')
    interpolated_sig = interpolator(np.arange(0,len(data_t_st_targ_ms)))
    std_env_p_val_interp.append(interpolated_sig)

std_env_p_val_interp = np.stack(std_env_p_val_interp)
interpolator = interpolate.interp1d(used_frequencies, std_env_p_val_interp, axis=0, kind='nearest', fill_value='extrapolate')
std_env_p_val_interp = interpolator(wanted_frequencies)


std_p = helper_functions.stepdown_p(std_stat, std_boot_stat)

std_p_val_arr = []
cnt = 0
for i in range(len(tf_waves_targ)):
    temp_arr = []
    for j in range(tf_waves_targ[i].shape[-1]):
        temp_arr.append(std_p[cnt])
        cnt += 1
    std_p_val_arr.append(temp_arr)

std_p_val_interp = []
for i in range(len(tf_waves_targ)):

    interpolator = interpolate.interp1d(coords_time_targ[i], std_p_val_arr[i], kind='nearest', fill_value='extrapolate')
    interpolated_sig = interpolator(np.arange(0,len(data_t_st_targ_ms)))
    std_p_val_interp.append(interpolated_sig)

std_p_val_interp = np.stack(std_p_val_interp)
interpolator = interpolate.interp1d(used_frequencies, std_p_val_interp, axis=0, kind='nearest', fill_value='extrapolate')
std_p_val_interp = interpolator(wanted_frequencies)

# %%
from matplotlib.patches import Patch

rect_patch = Patch(
    facecolor='none',
    edgecolor='limegreen',
    linewidth=1.5,
    label='p<0.05'
)

plt.rcParams['image.cmap'] = 'CMRmap'
fig, axs = plt.subplots(2, 2, figsize=(16, 9))
plt_header('Time-frequency resolved data: recorded, variable, and static bursts, subject ' + subject,
            use_suptitle=True, fontsize=18)

i,j=0,0
data_y = env_avg_out_interp
axs[i][j].set_title('A. |Avg(trials)| = average phase-locked response', fontsize=16)
img = axs[i][j].pcolormesh(data_t_st_targ_ms, wanted_frequencies, data_y, vmin=0, vmax=80)
axs[i][j].contour(data_t_st_targ_ms, wanted_frequencies, env_avg_p_val_interp, levels=[0.05],
                  colors='limegreen', linewidths=1)
axs[i][j].grid(visible=True, which='both', c='gray')
axs[i][j].set_ylabel('frequency [Hz]', fontsize=14)
#axs[i][j].set_xlabel('trial time [ms]', fontsize=14)
clb = fig.colorbar(img, extend='max')
clb.set_label('SNNR [fT/fT]', fontsize=14)
clb.ax.tick_params(labelsize=12)
axs[i][j].set_yscale('log')
axs[i][j].get_yaxis().set_minor_formatter(plticker.ScalarFormatter())
axs[i][j].get_yaxis().set_major_formatter(plticker.ScalarFormatter())
axs[i][j].tick_params(axis='x', which='both', labelbottom=False)
axs[i][j].set_ylim([200,2000])
axs[i][j].set_xlim([0,100])
axs[i][j].xaxis.set_major_locator(plticker.MultipleLocator(10))
axs[i][j].tick_params(which='both', labelsize=12)
axs[i][j].legend(handles=[rect_patch], fontsize=12, labelcolor='white',
                 framealpha=0.3, loc='upper right', bbox_to_anchor=(1.012, 1.02))

i,j=0,1
data_y = 20*np.log10(np.clip(std_out_interp, sys.float_info.min, None))
axs[i][j].set_title('B. StDev(Re(trials)) = single-trial variability', fontsize=16)
img = axs[i][j].pcolormesh(data_t_st_targ_ms, wanted_frequencies, data_y, vmin=0, vmax=4)
axs[i][j].contour(data_t_st_targ_ms, wanted_frequencies, std_p_val_interp, levels=[0.05],
                  colors='limegreen', linewidths=1)
axs[i][j].grid(visible=True, which='both', c='gray')
#axs[i][j].set_ylabel('frequency [Hz]', fontsize=14)
#axs[i][j].set_xlabel('trial time [ms]', fontsize=14)
clb = fig.colorbar(img, extend='both')
clb.set_label('SNNR [dB]', fontsize=14)
clb.ax.tick_params(labelsize=12)
axs[i][j].set_yscale('log')
axs[i][j].get_yaxis().set_minor_formatter(plticker.ScalarFormatter())
axs[i][j].get_yaxis().set_major_formatter(plticker.ScalarFormatter())
axs[i][j].tick_params(axis='y', which='both', labelleft=False)
axs[i][j].tick_params(axis='x', which='both', labelbottom=False)
axs[i][j].set_ylim([200,2000])
axs[i][j].set_xlim([0,100])
axs[i][j].xaxis.set_major_locator(plticker.MultipleLocator(10))
axs[i][j].tick_params(which='both', labelsize=12)
axs[i][j].legend(handles=[rect_patch], fontsize=12, labelcolor='white',
                 framealpha=0.3, loc='upper right', bbox_to_anchor=(1.012, 1.02))


i,j=1,0
data_y = 20*np.log10(np.clip(avg_env_out_interp, sys.float_info.min, None))
axs[i][j].set_title('C. Avg(|trials|) = average phase-insensitive response', fontsize=16)
img = axs[i][j].pcolormesh(data_t_st_targ_ms, wanted_frequencies, data_y, vmin=0, vmax=8)
axs[i][j].contour(data_t_st_targ_ms, wanted_frequencies, avg_env_p_val_interp, levels=[0.05],
                  colors='limegreen', linewidths=1)
axs[i][j].grid(visible=True, which='both', c='gray')
axs[i][j].set_ylabel('frequency [Hz]', fontsize=14)
axs[i][j].set_xlabel('trial time [ms]', fontsize=14)
clb = fig.colorbar(img, extend='both')
clb.set_label('SNNR [dB]', fontsize=14)
clb.ax.tick_params(labelsize=12)
clb.ax.yaxis.set_major_locator(plticker.MultipleLocator(1))
axs[i][j].set_yscale('log')
axs[i][j].get_yaxis().set_minor_formatter(plticker.ScalarFormatter())
axs[i][j].get_yaxis().set_major_formatter(plticker.ScalarFormatter())
axs[i][j].set_ylim([200,2000])
axs[i][j].set_xlim([0,100])
axs[i][j].xaxis.set_major_locator(plticker.MultipleLocator(10))
axs[i][j].tick_params(which='both', labelsize=12)
axs[i][j].legend(handles=[rect_patch], fontsize=12, labelcolor='white',
                 framealpha=0.3, loc='upper right', bbox_to_anchor=(1.012, 1.02))


i,j=1,1
data_y = 20*np.log10(np.clip(std_env_out_interp, sys.float_info.min, None))
axs[i][j].set_title('D. StDev(|trials|) = single-trial magnitude variability', fontsize=16)
img = axs[i][j].pcolormesh(data_t_st_targ_ms, wanted_frequencies, data_y, vmin=0, vmax=6)
axs[i][j].contour(data_t_st_targ_ms, wanted_frequencies, std_env_p_val_interp, levels=[0.05],
                  colors='limegreen', linewidths=1)
axs[i][j].grid(visible=True, which='both', c='gray')
#axs[i][j].set_ylabel('frequency [Hz]', fontsize=14)
axs[i][j].set_xlabel('trial time [ms]', fontsize=14)
clb = fig.colorbar(img, extend='both')
clb.set_label('SNNR [dB]', fontsize=14)
clb.ax.tick_params(labelsize=12)
axs[i][j].set_yscale('log')
axs[i][j].get_yaxis().set_minor_formatter(plticker.ScalarFormatter())
axs[i][j].get_yaxis().set_major_formatter(plticker.ScalarFormatter())
axs[i][j].tick_params(axis='y', which='both', labelleft=False)
axs[i][j].set_ylim([200,2000])
axs[i][j].set_xlim([0,100])
axs[i][j].xaxis.set_major_locator(plticker.MultipleLocator(10))
axs[i][j].tick_params(which='both', labelsize=12)
axs[i][j].legend(handles=[rect_patch], fontsize=12, labelcolor='white',
                 framealpha=0.3, loc='upper right', bbox_to_anchor=(1.012, 1.02))

for k in range(4):

    if(k==0):
        i,j = 0,0
    elif(k==1):
        i,j = 0,1
    elif(k==2):
        i,j = 1,0
    elif(k==3):
        i,j = 1,1

    axs[i][j].text(0.15, 0.86, 'rec.', fontsize=16, ha='center', va='center',
                   transform=axs[i][j].transAxes, c='white')

    axs[i][j].text(0.45, 0.86, 'var.', fontsize=16,
                ha='center', va='center', transform=axs[i][j].transAxes, c='white')

    axs[i][j].text(0.75, 0.86, 'stat.', fontsize=16,
                ha='center', va='center', transform=axs[i][j].transAxes, c='white')

fig.tight_layout()
plt_show_save_fig(subject+'_long')

# %%
from matplotlib.patches import Patch

rect_patch = Patch(
    facecolor='none',
    edgecolor='limegreen',
    linewidth=1.5,
    label='p<0.05'
)

plt.rcParams['image.cmap'] = 'CMRmap'
fig, axs = plt.subplots(2, 2, figsize=(16, 9))
plt_header('Time-frequency resolved data: recorded and variable bursts, subject ' + subject,
            use_suptitle=True, fontsize=18)

i,j=0,0
data_y = env_avg_out_interp
axs[i][j].set_title('A. |Avg(trials)| = average phase-locked response', fontsize=16)
img = axs[i][j].pcolormesh(data_t_st_targ_ms, wanted_frequencies, data_y, vmin=0, vmax=80)
axs[i][j].contour(data_t_st_targ_ms, wanted_frequencies, env_avg_p_val_interp, levels=[0.05],
                  colors='limegreen', linewidths=1)
axs[i][j].grid(visible=True, which='both', c='gray')
axs[i][j].set_ylabel('frequency [Hz]', fontsize=14)
#axs[i][j].set_xlabel('trial time [ms]', fontsize=14)
clb = fig.colorbar(img, extend='max')
clb.set_label('SNNR [fT/fT]', fontsize=14)
clb.ax.tick_params(labelsize=12)
axs[i][j].set_yscale('log')
axs[i][j].get_yaxis().set_minor_formatter(plticker.ScalarFormatter())
axs[i][j].get_yaxis().set_major_formatter(plticker.ScalarFormatter())
axs[i][j].tick_params(axis='x', which='both', labelbottom=False)
axs[i][j].set_ylim([200,2000])
axs[i][j].set_xlim([0,70])
axs[i][j].xaxis.set_major_locator(plticker.MultipleLocator(10))
axs[i][j].tick_params(which='both', labelsize=12)
axs[i][j].legend(handles=[rect_patch], fontsize=12, labelcolor='white',
                 framealpha=0.3, loc='upper right', bbox_to_anchor=(1.012, 1.02))

i,j=0,1
data_y = 20*np.log10(np.clip(std_out_interp, sys.float_info.min, None))
axs[i][j].set_title('B. StDev(Re(trials)) = single-trial variability', fontsize=16)
img = axs[i][j].pcolormesh(data_t_st_targ_ms, wanted_frequencies, data_y, vmin=0, vmax=4)
axs[i][j].contour(data_t_st_targ_ms, wanted_frequencies, std_p_val_interp, levels=[0.05],
                  colors='limegreen', linewidths=1)
axs[i][j].grid(visible=True, which='both', c='gray')
#axs[i][j].set_ylabel('frequency [Hz]', fontsize=14)
#axs[i][j].set_xlabel('trial time [ms]', fontsize=14)
clb = fig.colorbar(img, extend='both')
clb.set_label('SNNR [dB]', fontsize=14)
clb.ax.tick_params(labelsize=12)
axs[i][j].set_yscale('log')
axs[i][j].get_yaxis().set_minor_formatter(plticker.ScalarFormatter())
axs[i][j].get_yaxis().set_major_formatter(plticker.ScalarFormatter())
axs[i][j].tick_params(axis='y', which='both', labelleft=False)
axs[i][j].tick_params(axis='x', which='both', labelbottom=False)
axs[i][j].set_ylim([200,2000])
axs[i][j].set_xlim([0,70])
axs[i][j].xaxis.set_major_locator(plticker.MultipleLocator(10))
axs[i][j].tick_params(which='both', labelsize=12)
axs[i][j].legend(handles=[rect_patch], fontsize=12, labelcolor='white',
                 framealpha=0.3, loc='upper right', bbox_to_anchor=(1.012, 1.02))


i,j=1,0
data_y = 20*np.log10(np.clip(avg_env_out_interp, sys.float_info.min, None))
axs[i][j].set_title('C. Avg(|trials|) = average phase-insensitive response', fontsize=16)
img = axs[i][j].pcolormesh(data_t_st_targ_ms, wanted_frequencies, data_y, vmin=0, vmax=8)
axs[i][j].contour(data_t_st_targ_ms, wanted_frequencies, avg_env_p_val_interp, levels=[0.05],
                  colors='limegreen', linewidths=1)
axs[i][j].grid(visible=True, which='both', c='gray')
axs[i][j].set_ylabel('frequency [Hz]', fontsize=14)
axs[i][j].set_xlabel('trial time [ms]', fontsize=14)
clb = fig.colorbar(img, extend='both')
clb.set_label('SNNR [dB]', fontsize=14)
clb.ax.tick_params(labelsize=12)
clb.ax.yaxis.set_major_locator(plticker.MultipleLocator(1))
axs[i][j].set_yscale('log')
axs[i][j].get_yaxis().set_minor_formatter(plticker.ScalarFormatter())
axs[i][j].get_yaxis().set_major_formatter(plticker.ScalarFormatter())
axs[i][j].set_ylim([200,2000])
axs[i][j].set_xlim([0,70])
axs[i][j].xaxis.set_major_locator(plticker.MultipleLocator(10))
axs[i][j].tick_params(which='both', labelsize=12)
axs[i][j].legend(handles=[rect_patch], fontsize=12, labelcolor='white',
                 framealpha=0.3, loc='upper right', bbox_to_anchor=(1.012, 1.02))


i,j=1,1
data_y = 20*np.log10(np.clip(std_env_out_interp, sys.float_info.min, None))
axs[i][j].set_title('D. StDev(|trials|) = single-trial magnitude variability', fontsize=16)
img = axs[i][j].pcolormesh(data_t_st_targ_ms, wanted_frequencies, data_y, vmin=0, vmax=6)
axs[i][j].contour(data_t_st_targ_ms, wanted_frequencies, std_env_p_val_interp, levels=[0.05],
                  colors='limegreen', linewidths=1)
axs[i][j].grid(visible=True, which='both', c='gray')
#axs[i][j].set_ylabel('frequency [Hz]', fontsize=14)
axs[i][j].set_xlabel('trial time [ms]', fontsize=14)
clb = fig.colorbar(img, extend='both')
clb.set_label('SNNR [dB]', fontsize=14)
clb.ax.tick_params(labelsize=12)
axs[i][j].set_yscale('log')
axs[i][j].get_yaxis().set_minor_formatter(plticker.ScalarFormatter())
axs[i][j].get_yaxis().set_major_formatter(plticker.ScalarFormatter())
axs[i][j].tick_params(axis='y', which='both', labelleft=False)
axs[i][j].set_ylim([200,2000])
axs[i][j].set_xlim([0,70])
axs[i][j].xaxis.set_major_locator(plticker.MultipleLocator(10))
axs[i][j].tick_params(which='both', labelsize=12)
axs[i][j].legend(handles=[rect_patch], fontsize=12, labelcolor='white',
                 framealpha=0.3, loc='upper right', bbox_to_anchor=(1.012, 1.02))

for k in range(4):

    if(k==0):
        i,j = 0,0
    elif(k==1):
        i,j = 0,1
    elif(k==2):
        i,j = 1,0
    elif(k==3):
        i,j = 1,1

    axs[i][j].text(0.22, 0.86, 'rec.', fontsize=16, ha='center', va='center',
                   transform=axs[i][j].transAxes, c='white')

    axs[i][j].text(0.65, 0.86, 'var.', fontsize=16,
                ha='center', va='center', transform=axs[i][j].transAxes, c='white')

    #axs[i][j].text(0.75, 0.86, 'stat.', fontsize=16,
    #            ha='center', va='center', transform=axs[i][j].transAxes, c='white')

fig.tight_layout()
plt_show_save_fig(subject+'_short')


