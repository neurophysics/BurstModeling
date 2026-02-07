# %%
import os
import numpy as np
import pandas as pd
import pingouin as pg
import matplotlib.pyplot as plt
from scipy.stats import shapiro
from scipy.stats import binomtest
import matplotlib.ticker as plticker

plt.rcParams['figure.figsize'] = [12, 6]
plt.rcParams['savefig.dpi'] = 600

# %%
data_folder = '../Results/burst_modeling/statistical_analysis_proper'

# %%
df_S1 = pd.read_csv(os.path.join(data_folder, 'S1', 'S1_results.csv'))
df_S2 = pd.read_csv(os.path.join(data_folder, 'S2', 'S2_results.csv'))
df_S3 = pd.read_csv(os.path.join(data_folder, 'S3', 'S3_results.csv'))
df_S4 = pd.read_csv(os.path.join(data_folder, 'S4', 'S4_results.csv'))
df_S5 = pd.read_csv(os.path.join(data_folder, 'S5', 'S5_results.csv'))

df_all = pd.concat([df_S1, df_S2, df_S3, df_S4, df_S5], ignore_index=True)

subj_fold_col = df_all['subject']+'_f'+df_all['fold'].astype('string')
df_all.insert(2, 'subj_fold', subj_fold_col)
df_all = df_all.round(3)

# %%
df_pivot = df_all.pivot(
    index=['subject', 'fold'],
    columns='opt_params',
    values='test_loss'
).reset_index()

df_pivot.columns.name = None

# %%
mean_row = {col: None for col in df_pivot.columns}
mean_row['subject'] = 'mean'
mean_row['fold'] = '------------'
mean_row[0] = df_pivot[0].mean()
mean_row[2] = df_pivot[2].mean()
mean_row[4] = df_pivot[4].mean()
mean_row[6] = df_pivot[6].mean()

# %%
total_row = {col: None for col in df_pivot.columns}
total_row['subject'] = 'total'
total_row['fold'] = 'better'
total_row[0] = np.nan
total_row[2] = (df_pivot[2] < df_pivot[0]).sum()
total_row[4] = (df_pivot[4] < df_pivot[2]).sum()
total_row[6] = (df_pivot[6] < df_pivot[4]).sum()

# %%
rows_n = len(df_pivot)

p_binom_row = {col: None for col in df_pivot.columns}
p_binom_row['subject'] = 'p-val'
p_binom_row['fold'] = 'binomial'
p_binom_row[0] = np.nan
p_binom_row[2] = binomtest(total_row[2], rows_n, p=0.5, alternative="two-sided").pvalue
p_binom_row[4] = binomtest(total_row[4], rows_n, p=0.5, alternative="two-sided").pvalue
p_binom_row[6] = binomtest(total_row[6], rows_n, p=0.5, alternative="two-sided").pvalue

# %%
p_wilcox_row = {col: None for col in df_pivot.columns}
p_wilcox_row['subject'] = 'p-val'
p_wilcox_row['fold'] = 'Wilcoxon'
p_wilcox_row[0] = np.nan
p_wilcox_row[2] = pg.wilcoxon(df_pivot[2], df_pivot[0], alternative='less')['p-val'].values[0]
p_wilcox_row[4] = pg.wilcoxon(df_pivot[4], df_pivot[2], alternative='less')['p-val'].values[0]
p_wilcox_row[6] = pg.wilcoxon(df_pivot[6], df_pivot[4], alternative='less')['p-val'].values[0]

# %%
def highlight_opt_params_diff(s):

    styles = ["" for _ in s.index]

    # Check if this is the TOTAL row
    if s['subject'] == 'total' or (s['subject'] == 'p-val' and s['fold'] == 'binomial'):
        for i in [2,4,6]:
            if(p_binom_row[i] < 0.05):
                styles[s.index.get_loc(i)] = "background-color: green"
            else:
                styles[s.index.get_loc(i)] = "background-color: red"
    elif (s['subject'] == 'p-val' and s['fold'] == 'Wilcoxon'):
        for i in [2,4,6]:
            if(p_wilcox_row[i] < 0.05):
                styles[s.index.get_loc(i)] = "background-color: green"
            else:
                styles[s.index.get_loc(i)] = "background-color: red"
    else:

        if(s['subject'] == 'mean'):
                color_red = "background-color: red"
                color_green = "background-color: green"
        else:
                color_red = "background-color: darkred"
                color_green = "background-color: darkgreen"  
          
        if s[2] < s[0]:
            styles[s.index.get_loc(2)] = color_green
        else:
            styles[s.index.get_loc(2)] = color_red

        if s[4] < s[2]:
            styles[s.index.get_loc(4)] = color_green
        else:
            styles[s.index.get_loc(4)] = color_red

        if s[6] < s[4]:
            styles[s.index.get_loc(6)] = color_green
        else:
            styles[s.index.get_loc(6)] = color_red

    return styles

# %%
df_pivot_res = pd.concat([df_pivot, pd.DataFrame([mean_row]), pd.DataFrame([total_row]),
                          pd.DataFrame([p_binom_row]), pd.DataFrame([p_wilcox_row])], ignore_index=True)
df_pivot_res.style.apply(highlight_opt_params_diff,
                         axis=1).format({0: '{:.3f}', 2: '{:.3f}', 4: '{:.3f}', 6: '{:.3f}'})

# %%
# test for normality of the folds differences
# if p > 0.05 we assume the distribution is normal

print('%f' % shapiro(df_pivot[0] - df_pivot[2]).pvalue)
print('%f' % shapiro(df_pivot[2] - df_pivot[4]).pvalue)
print('%f' % shapiro(df_pivot[4] - df_pivot[6]).pvalue)

# %%
df_pivot_mean = df_pivot.groupby('subject').mean().reset_index().drop('fold', axis=1)

mean_row_mean = df_pivot[[0, 2, 4, 6]].mean().to_frame().T
mean_row_mean['subject'] = 'mean'
df_pivot_mean2 = pd.concat([df_pivot_mean, mean_row_mean], ignore_index=True)

df_pivot_mean2.style.apply(highlight_opt_params_diff, axis=1).format(
    {0: '{:.3f}', 2: '{:.3f}', 4: '{:.3f}', 6: '{:.3f}'})

# %%
# Wilcoxon signed-rank test

print(pg.wilcoxon(df_pivot_mean[2], df_pivot_mean[0], alternative='less').round(6))
print(pg.wilcoxon(df_pivot_mean[4], df_pivot_mean[2], alternative='less').round(6))
print(pg.wilcoxon(df_pivot_mean[6], df_pivot_mean[4], alternative='less').round(6))

# %%
# test for normality of the folds differences
# if p > 0.05 we assume the distribution is normal

print('%f' % shapiro(df_pivot_mean[0] - df_pivot_mean[2]).pvalue)
print('%f' % shapiro(df_pivot_mean[2] - df_pivot_mean[4]).pvalue)
print('%f' % shapiro(df_pivot_mean[4] - df_pivot_mean[6]).pvalue)

# %%
# T-test for pairs

print(pg.ttest(df_pivot_mean[2], df_pivot_mean[0], paired=True, alternative='less')['p-val'].round(6))
print(pg.ttest(df_pivot_mean[4], df_pivot_mean[2], paired=True, alternative='less')['p-val'].round(6))
print(pg.ttest(df_pivot_mean[6], df_pivot_mean[4], paired=True, alternative='less')['p-val'].round(6))

# %%
# Observed variability explanation [%]
# static burst model -> 0% of variability explained
# loss = 0 -> 100% of variability explained

df_pivot_mean_expl = df_pivot_mean2.copy()

df_pivot_mean_expl[6] /= df_pivot_mean_expl[0]
df_pivot_mean_expl[4] /= df_pivot_mean_expl[0]
df_pivot_mean_expl[2] /= df_pivot_mean_expl[0]
df_pivot_mean_expl[0] /= df_pivot_mean_expl[0]

df_pivot_mean_expl[6] = (1-df_pivot_mean_expl[6])*100
df_pivot_mean_expl[4] = (1-df_pivot_mean_expl[4])*100
df_pivot_mean_expl[2] = (1-df_pivot_mean_expl[2])*100
df_pivot_mean_expl[0] = (1-df_pivot_mean_expl[0])*100

df_pivot_mean_expl.style.format({0: '{:.3f}', 2: '{:.3f}', 4: '{:.3f}', 6: '{:.3f}'})

# %%
fig, axs = plt.subplots()
plt.title('Models comparison and observed variability explanation', fontsize=14)
x_labels = ['model A\n0 parameters', 'model D\n2 parameters', 'model E\n4 parameters', 'model F\n6 parameters']

for i in range(len(df_pivot_mean_expl)):
    y_values_test = df_pivot_mean_expl.iloc[i][[0,2,4,6]]
    subj_i = df_pivot_mean_expl['subject'][i]
    if(subj_i == 'mean'):
        axs.plot(x_labels, y_values_test , 'o-', label=subj_i, c='0.2', alpha=0.8)
    else:
        axs.plot(x_labels, y_values_test , 'o-', label=subj_i, alpha=0.8)

axs.grid(axis='y')
axs.margins(x=0.2)
axs.legend(loc='upper left')
axs.yaxis.set_major_locator(plticker.MultipleLocator(10))
axs.set_ylim(-5, 105)
axs.set_ylabel('% of observed variability explained')
plt.savefig(os.path.join(data_folder, 'variability explanation'), bbox_inches='tight')

# %%
# div_steep - division curve steepness
# div_pos - division curve center position
# ecavs - earlier component amplitude variability scaler
# lcavs - later component amplitude variability scaler
# eclvs - earlier component latency variability scaler
# lclvs - later component latency variability scaler

pd.set_option('display.max_rows', 20)
df_opt_6 = df_all[df_all['opt_params']==6]
df_opt_6.round(3)

# %%
print(pg.wilcoxon(df_opt_6['ecavs'], df_opt_6['lcavs']).round(6))
print(pg.wilcoxon(df_opt_6['eclvs'], df_opt_6['lclvs']).round(6))

# %%
df_opt_6 = df_all[df_all['opt_params']==6]

mean_row_6 = df_opt_6.drop(['subject', 'fold', 'subj_fold', 'opt_params'], axis=1).mean().to_frame().T
mean_row_6['subject'] = 'mean'

df_opt_6_mean = df_opt_6.drop(['subj_fold', 'fold', 'opt_params'], axis=1).groupby('subject').mean().reset_index()
df_opt_6_mean2 = pd.concat([df_opt_6_mean, mean_row_6], ignore_index=True)

df_opt_6_mean2.round(3)

# %%
# test for normality of the folds differences
# if p > 0.05 we assume the distribution is normal

print('%f' % shapiro(df_opt_6_mean['lcavs'] - df_opt_6_mean['ecavs']).pvalue)
print('%f' % shapiro(df_opt_6_mean['lclvs'] - df_opt_6_mean['eclvs']).pvalue)

# %%
# T-test for pairs

print(pg.ttest(df_opt_6_mean['lcavs'], df_opt_6_mean['ecavs'], paired=True, alternative='greater')['p-val'].round(6))
print(pg.ttest(df_opt_6_mean['lclvs'], df_opt_6_mean['eclvs'], paired=True, alternative='greater')['p-val'].round(6))

# %%
df_opt_4 = df_all[df_all['opt_params']==4]

mean_row_4 = df_opt_4.drop(['subject', 'fold', 'subj_fold', 'opt_params'], axis=1).mean().to_frame().T
mean_row_4['subject'] = 'mean'

df_opt_4_mean = df_opt_4.drop(['subj_fold', 'fold', 'opt_params'], axis=1).groupby('subject').mean().reset_index()
df_opt_4_mean2 = pd.concat([df_opt_4_mean, mean_row_4], ignore_index=True)

df_opt_4_mean2[['ecavs', 'eclvs']] = '-----'
df_opt_4_mean2.round(3)

# %%
df_opt_6 = df_all[df_all['opt_params']==6]
mean_row_6 = df_opt_6.drop(['subject', 'fold', 'subj_fold'], axis=1).mean().to_frame().T.round(3)
mean_row_6['opt_params'] = 6

df_opt_4 = df_all[df_all['opt_params']==4]
mean_row_4 = df_opt_4.drop(['subject', 'fold', 'subj_fold'], axis=1).mean().to_frame().T.round(3)
mean_row_4[['ecavs', 'eclvs']] = '-----'
mean_row_4['opt_params'] = 4

df_opt_2 = df_all[df_all['opt_params']==2]
mean_row_2 = df_opt_2.drop(['subject', 'fold', 'subj_fold'], axis=1).mean().to_frame().T.round(3)
mean_row_2['opt_params'] = 2
mean_row_2[['div_steep', 'div_pos', 'ecavs', 'eclvs']] = '-----'

df_opt_0 = df_all[df_all['opt_params']==0]
mean_row_0 = df_opt_0.drop(['subject', 'fold', 'subj_fold'], axis=1).mean().to_frame().T.round(3)
mean_row_0[['div_steep', 'div_pos', 'ecavs', 'lcavs', 'eclvs', 'lclvs']] = '-----'
mean_row_0['opt_params'] = 0

# %%
pd.concat([mean_row_0, mean_row_2, mean_row_4, mean_row_6], ignore_index=True)


