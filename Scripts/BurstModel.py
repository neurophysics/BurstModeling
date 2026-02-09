"""
BurstModel.py
Lukasz Radzinski
Charité Neurophysics Group, Berlin
Burst variability model
"""

import numpy as np
import tensorflow as tf
from scipy import signal as sig


class BurstModel:

    # Burst variability model
    # optimize variability parameters
    # to get the best fit between
    # simulated and recorded bursts

    # optimized parameters:
    # div_steep - division curve steepness
    # div_pos - division curve center position
    # ecavs - early component amplitude variability scaler
    # eclvs - early component latency variability scaler
    # lcavs - late component amplitude variability scaler
    # lclvs - late component latency variability scaler

    def __init__(self, data_t_medium_ms, sigma_medium_trials, burst_win_ms, sigma_sim_offset_ms, srate, seed=60):

        self.data_t_medium_ms_tf = tf.constant(data_t_medium_ms) # time axis
        self.sigma_medium_trials_tf = tf.constant(sigma_medium_trials) # single trials
        self.sigma_medium_trials_er_tf = tf.constant(sigma_medium_trials.mean(-1)) # evoked response
    
        self.burst_win_ms = burst_win_ms # time window for recorded sigma burst
        self.burst_mt_smpl = [np.nonzero(data_t_medium_ms == burst_win_ms[0])[0][0],
                              np.nonzero(data_t_medium_ms == burst_win_ms[1])[0][0]]
        self.burst_mt_smpl = np.array(self.burst_mt_smpl) # time samples for recorded sigma burst

        # time samples for simulated sigma burst
        self.burst_sim_mt_smpl = np.array(self.burst_mt_smpl)+sigma_sim_offset_ms*srate//1000

        # Tukey window to taper the edges of the burst for the evoked response and loss calculations
        self.tukey_win_er = tf.constant(sig.windows.tukey(np.diff(self.burst_mt_smpl)[0], 0.2))
        self.tukey_win_loss = tf.constant(sig.windows.tukey(np.diff(self.burst_mt_smpl)[0], 0.4))

        n_trials = sigma_medium_trials.shape[-1] # number of trials

        # set Gaussian distributions for the burst variability
        rng = np.random.default_rng(seed=seed)
        self.randn_lcav = np.clip(rng.standard_normal(n_trials), -3, 3) # late component amplitude variability
        self.randn_lclv = np.clip(rng.standard_normal(n_trials), -3, 3) # late component latency variability

        self.randn_ecav = np.clip(rng.standard_normal(n_trials), -3, 3) # early component amplitude variability
        self.randn_eclv = np.clip(rng.standard_normal(n_trials), -3, 3) # early component latency variability

        # latency variability constants
        self.cmp_offset = 100
        comp_len = np.diff(self.burst_mt_smpl)[0]

        # constants for latency variability interpolation
        self.cols_range = tf.range(comp_len, dtype='float64')
        self.tf_offset_zeros = tf.zeros(self.cmp_offset, dtype='complex128')
        self.tf_trials_ones = tf.ones([n_trials, 1], 'complex128')
        self.rows = tf.range(n_trials)
        self.rows = tf.repeat(self.rows, comp_len)
        self.rows = tf.reshape(self.rows, (-1, comp_len))

    # calculate output of the simulated burst model
    def calculate_model_output_raw(self, div_steep, div_pos_ms, ecavs, eclvs, lcavs, lclvs):

        # calculate division curve
        exponent = (self.data_t_medium_ms_tf - div_pos_ms) * div_steep
        division_curve = tf.math.sigmoid(exponent)


        # extract late component
        late_comp_sigma_er_long = self.sigma_medium_trials_er_tf*tf.cast(division_curve, 'complex128')
        late_comp_sigma_raw_er = late_comp_sigma_er_long[self.burst_mt_smpl[0]:self.burst_mt_smpl[1]]
        late_comp_sigma_er = late_comp_sigma_raw_er*tf.cast(self.tukey_win_er, 'complex128')

        # add latency variability using interpolation method
        late_comp_sigma_out = self.add_latency_variability(late_comp_sigma_raw_er, self.randn_lclv, lclvs)

        # calculate scaler for late component base amplitude
        # to compensate the added latency variability effect on the amplitude
        # late component amplitude base scaler
        lcabs = (tf.reduce_mean(tf.math.abs(late_comp_sigma_er)) / 
                tf.reduce_mean(tf.math.abs(tf.reduce_mean(late_comp_sigma_out, axis=0))))

        # calculate amplitude of late component
        lcav_comb = self.randn_lcav*lcavs
        late_comp_ampl = lcabs+lcav_comb

        # prevent the amplitude to go below 0
        late_comp_ampl = tf.math.softplus(late_comp_ampl*5)/5
        late_comp_ampl = tf.cast(late_comp_ampl, 'complex128')

        # add amplitude variability
        late_comp_sigma_out = tf.transpose(tf.transpose(late_comp_sigma_out) * late_comp_ampl)


        # extract early component
        early_comp_sigma_er_long = self.sigma_medium_trials_er_tf - late_comp_sigma_er_long
        early_comp_sigma_raw_er = early_comp_sigma_er_long[self.burst_mt_smpl[0]:self.burst_mt_smpl[1]]
        early_comp_sigma_er = early_comp_sigma_raw_er*tf.cast(self.tukey_win_er, 'complex128')

        # add latency variability using interpolation method
        early_comp_sigma_out = self.add_latency_variability(early_comp_sigma_raw_er, self.randn_eclv, eclvs)

        # calculate scaler for early component base amplitude
        # to compensate the added latency variability effect on the amplitude
        # early component amplitude base scaler
        ecabs = (tf.reduce_mean(tf.math.abs(early_comp_sigma_er)) /
                tf.reduce_mean(tf.math.abs(tf.reduce_mean(early_comp_sigma_out, axis=0))))

        # calculate amplitude of early component
        early_comp_ampl = ecabs+self.randn_ecav*ecavs

        # prevent the amplitude to go below 0
        early_comp_ampl = tf.math.softplus(early_comp_ampl*5)/5
        early_comp_ampl = tf.cast(early_comp_ampl, 'complex128')

        # add amplitude variability
        early_comp_sigma_out = tf.transpose(tf.transpose(early_comp_sigma_out) * early_comp_ampl)


        # combine two components of simulated burst
        sigma_bursts_sim = tf.transpose(early_comp_sigma_out + late_comp_sigma_out)

        # superpose the simulated burst with ongoing noise of recorded single trials
        tf_zeros_1 = tf.zeros((self.burst_sim_mt_smpl[0], self.sigma_medium_trials_tf.shape[1]), dtype='complex128')
        tf_zeros_2 = tf.zeros((self.sigma_medium_trials_tf.shape[0] - self.burst_sim_mt_smpl[1],
                            self.sigma_medium_trials_tf.shape[1]), dtype='complex128')
        sigma_sim_medium_trials = tf.concat((tf_zeros_1, sigma_bursts_sim, tf_zeros_2), axis=0)
        sigma_sim_medium_trials = sigma_sim_medium_trials + self.sigma_medium_trials_tf

        # calculate loss between recorded and simulated bursts
        result = self.calculate_model_loss(sigma_sim_medium_trials)

        return result, division_curve, late_comp_sigma_er, lcabs, early_comp_sigma_er, ecabs, sigma_bursts_sim, sigma_sim_medium_trials

    # wrapper function to pass the arguments properly
    def calculate_model_output(self, model_variables):

        [div_steep, div_pos, ecavs, eclvs, lcavs, lclvs] = model_variables

        div_pos_ms = div_pos*(self.burst_win_ms[1]-self.burst_win_ms[0])+self.burst_win_ms[0]

        return self.calculate_model_output_raw(div_steep, div_pos_ms, ecavs, eclvs, lcavs, lclvs)

    # add latency variability and use interpolation to make latency shifts continuous
    def add_latency_variability(self, component, randn_nclv, nclvs):

        start_indices = self.cmp_offset+randn_nclv*nclvs
        start_indices = tf.clip_by_value(start_indices, 0, self.cmp_offset*2-1)
        
        component_long = tf.concat((self.tf_offset_zeros, component, self.tf_offset_zeros), axis=0)
        cols = start_indices[:, tf.newaxis] + self.cols_range
        cols_floor = tf.floor(cols)
        weights_ceil = cols - cols_floor
        weights_floor = 1.0 - weights_ceil
        weights_ceil = tf.cast(weights_ceil, 'complex128')
        weights_floor = tf.cast(weights_floor, 'complex128')

        new_comp = tf.tensordot(self.tf_trials_ones, [component_long], axes=1)

        indices_floor = tf.stack([self.rows, tf.cast(cols_floor, 'int32')], axis=-1)
        indices_ceil = tf.stack([self.rows, tf.cast(cols_floor, 'int32')+1], axis=-1)
        values_floor = tf.gather_nd(new_comp, indices_floor)
        values_ceil = tf.gather_nd(new_comp, indices_ceil)

        new_comp = values_floor * weights_floor + values_ceil * weights_ceil
        new_comp_out = new_comp*tf.cast(self.tukey_win_er, 'complex128')

        return new_comp_out


    # calculate loss function
    def calculate_model_loss(self, sigma_trials_sim, is_full_output=False):

        burst_mt_smpl = self.burst_mt_smpl
        burst_sim_mt_smpl = self.burst_sim_mt_smpl
        tukey_win_loss = self.tukey_win_loss

        sigma_trials_sim_real = tf.math.real(sigma_trials_sim)
        sigma_trials_sim_abs = tf.math.abs(sigma_trials_sim)


        # averaged response curves difference
        data_y = tf.math.reduce_mean(sigma_trials_sim_real, axis=-1)
        # extract recorded (physiological) response and normalize it
        avg_phys = data_y[burst_mt_smpl[0]:burst_mt_smpl[1]]
        avg_phys_mc = avg_phys
        avg_phys_tuk = avg_phys_mc*tukey_win_loss
        avg_phys_std = tf.math.reduce_std(avg_phys_tuk)
        avg_phys_norm = avg_phys_tuk/avg_phys_std
        # extract simulated response and normalize it regarding the recorded one
        avg_sim = data_y[burst_sim_mt_smpl[0]:burst_sim_mt_smpl[1]]
        avg_sim_mc = avg_sim
        avg_sim_tuk = avg_sim_mc*tukey_win_loss
        avg_sim_norm = avg_sim_tuk/avg_phys_std
        # calculate MSE (Mean Squared Error)
        mse_avg = tf.math.reduce_mean((avg_phys_norm - avg_sim_norm)**2)


        # standard deviation curves difference
        data_y = tf.math.reduce_std(sigma_trials_sim_real, axis=-1)
        # extract recorded (physiological) response and normalize it
        std_phys = data_y[burst_mt_smpl[0]:burst_mt_smpl[1]]
        std_phys_min = tf.math.reduce_min(std_phys)
        std_phys_mc = std_phys - std_phys_min
        std_phys_tuk = std_phys_mc*tukey_win_loss
        std_phys_std = tf.math.reduce_std(std_phys_tuk)
        std_phys_norm = std_phys_tuk/std_phys_std
        # extract simulated response and normalize it regarding the recorded one
        std_sim = data_y[burst_sim_mt_smpl[0]:burst_sim_mt_smpl[1]]
        std_sim_mc = std_sim - std_phys_min
        std_sim_tuk = std_sim_mc*tukey_win_loss
        std_sim_norm = std_sim_tuk/std_phys_std
        # calculate MSE (Mean Squared Error)
        mse_std = tf.math.reduce_mean((std_phys_norm - std_sim_norm)**2)


        # average of envelopes curves difference
        data_y = tf.math.reduce_mean(sigma_trials_sim_abs, axis=-1)
        # extract recorded (physiological) response and normalize it
        avg_env_phys = data_y[burst_mt_smpl[0]:burst_mt_smpl[1]]
        avg_env_phys_min = tf.math.reduce_min(avg_env_phys)
        avg_env_phys_mc = avg_env_phys - avg_env_phys_min
        avg_env_phys_tuk = avg_env_phys_mc*tukey_win_loss
        avg_env_phys_std = tf.math.reduce_std(avg_env_phys_tuk)
        avg_env_phys_norm = avg_env_phys_tuk/avg_env_phys_std
        # extract simulated response and normalize it regarding the recorded one
        avg_env_sim = data_y[burst_sim_mt_smpl[0]:burst_sim_mt_smpl[1]]
        avg_env_sim_mc = avg_env_sim - avg_env_phys_min
        avg_env_sim_tuk = avg_env_sim_mc*tukey_win_loss
        avg_env_sim_norm = avg_env_sim_tuk/avg_env_phys_std
        # calculate MSE (Mean Squared Error)
        mse_avg_env = tf.math.reduce_mean((avg_env_phys_norm - avg_env_sim_norm)**2)


        # standard deviation of envelopes curves difference
        data_y = tf.math.reduce_std(sigma_trials_sim_abs, axis=-1)
        # extract recorded (physiological) response and normalize it
        std_env_phys = data_y[burst_mt_smpl[0]:burst_mt_smpl[1]]
        std_env_phys_min = tf.math.reduce_min(std_env_phys)
        std_env_phys_mc = std_env_phys - std_env_phys_min
        std_env_phys_tuk = std_env_phys_mc*tukey_win_loss
        std_env_phys_std = tf.math.reduce_std(std_env_phys_tuk)
        std_env_phys_norm = std_env_phys_tuk/std_env_phys_std
        # extract simulated response and normalize it regarding the recorded one
        std_env_sim = data_y[burst_sim_mt_smpl[0]:burst_sim_mt_smpl[1]]
        std_env_sim_mc = std_env_sim - std_env_phys_min
        std_env_sim_tuk = std_env_sim_mc*tukey_win_loss
        std_env_sim_norm = std_env_sim_tuk/std_env_phys_std
        # calculate MSE (Mean Squared Error)
        mse_std_env = tf.math.reduce_mean((std_env_phys_norm - std_env_sim_norm)**2)

        # calculate final loss
        loss = mse_avg + mse_std + mse_avg_env + mse_std_env

        if(is_full_output):
            return ((avg_phys_norm, avg_sim_norm, avg_phys_std, 0),
                    (std_phys_norm, std_sim_norm, std_phys_std, std_phys_min),
                    (avg_env_phys_norm, avg_env_sim_norm, avg_env_phys_std, avg_env_phys_min),
                    (std_env_phys_norm, std_env_sim_norm, std_env_phys_std, std_env_phys_min),
                    (mse_avg, mse_std, mse_avg_env, mse_std_env, loss))
        else:
            return loss