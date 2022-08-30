

import pandas as pd
import seaborn as sns

from matplotlib import pyplot as plt
from scipy.ndimage.filters import gaussian_filter1d


def smooth(scalars, weight):  # Weight between 0 and 1
    last = scalars[0]  # First value in the plot (first timestep)
    smoothed = list()
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point  # Calculate smoothed value
        smoothed.append(smoothed_val)                        # Save it
        last = smoothed_val                                  # Anchor the last smoothed value

    return smoothed


# from pnplot import *

# dataset1_filename = 'run-ATC_GRL_Model_PPO_7-tag-rollout_ep_rew_mean.csv'
# dataset2_filename = 'run-ATC_GRL_Model_PPO_7-tag-train_entropy_loss.csv'

# dataset1_filename = 'PPO3_Reward.csv'
# dataset2_filename = 'PPO3_Entropy.csv'

dataset1_filename = './300k_rl_results/ep_mean_near_misses.csv'
dataset2_filename = './300k_grl_results/ep_mean_near_misses.csv'
dataset3_filename = './300k_rl_results/ep_mean_tasks_completed.csv'
dataset4_filename = './300k_grl_results/ep_mean_tasks_completed.csv'
dataset5_filename = './300k_rl_results/ep_mean_total_delay.csv'
dataset6_filename = './300k_grl_results/ep_mean_total_delay.csv'
dataset7_filename = './300k_rl_results/ep_rew_mean.csv'
dataset8_filename = './300k_grl_results/ep_rew_mean.csv'
dataset9_filename = './300k_rl_results/mean_step_time.csv'
dataset10_filename = './300k_grl_results/mean_step_time.csv'
dataset11_filename = './300k_rl_results/train_entropy_loss.csv'
dataset12_filename = './300k_grl_results/train_entropy_loss.csv'
dataset13_filename = './300k_rl_results/train_policy_gradient_loss.csv'
dataset14_filename = './300k_grl_results/train_policy_gradient_loss.csv'

dataframe1 = pd.read_csv(open(dataset1_filename, 'rb'))
dataframe2 = pd.read_csv(open(dataset2_filename, 'rb'))
dataframe3 = pd.read_csv(open(dataset3_filename, 'rb'))
dataframe4 = pd.read_csv(open(dataset4_filename, 'rb'))
dataframe5 = pd.read_csv(open(dataset5_filename, 'rb'))
dataframe6 = pd.read_csv(open(dataset6_filename, 'rb'))
dataframe7 = pd.read_csv(open(dataset7_filename, 'rb'))
dataframe8 = pd.read_csv(open(dataset8_filename, 'rb'))
dataframe9 = pd.read_csv(open(dataset9_filename, 'rb'))
dataframe10 = pd.read_csv(open(dataset10_filename, 'rb'))
dataframe11 = pd.read_csv(open(dataset11_filename, 'rb'))
dataframe12 = pd.read_csv(open(dataset12_filename, 'rb'))
dataframe13 = pd.read_csv(open(dataset13_filename, 'rb'))
dataframe14 = pd.read_csv(open(dataset14_filename, 'rb'))

fig = plt.figure(figsize=(10,8))
axis = fig.add_subplot(221)
#Mission Time Figure

dataframe1 = dataframe1.ewm(alpha = 0.2).mean()
dataframe2 = dataframe2.ewm(alpha = 0.2).mean()
sns.lineplot(x="Episode", y="Value", data=dataframe1,label='rl agent')
sns.lineplot(x="Episode", y="Value", data=dataframe2,label='grl agent')


# plt.plot(x="Step", y="Value", hue="Value", data=dataframe)
plt.xlabel('Episodes')
plt.ylabel('Near Misses')
# plt.title('GRL Agent')
# plt.show()

axis2 = fig.add_subplot(222)
dataframe3 = dataframe3.ewm(alpha = 0.2).mean()
dataframe4 = dataframe4.ewm(alpha = 0.2).mean()
sns.lineplot(x="Episode", y="Value", data=dataframe3,label='rl agent')
sns.lineplot(x="Episode", y="Value", data=dataframe4,label='grl agent')
# plt.plot(x="Step", y="Value", hue="Value", data=dataframe)
plt.xlabel('Episodes')
plt.ylabel('Tasks Completed')
# plt.title('GRL Agent')
# plt.show()

axis3 = fig.add_subplot(223)
dataframe5 = dataframe5.ewm(alpha = 0.2).mean()
dataframe6 = dataframe6.ewm(alpha = 0.2).mean()
sns.lineplot(x="Episode", y="Value", data=dataframe5,label='rl agent')
sns.lineplot(x="Episode", y="Value", data=dataframe6,label='grl agent')
# plt.plot(x="Step", y="Value", hue="Value", data=dataframe)
plt.xlabel('Episodes')
plt.ylabel('Total Delay')
plt.legend()
# plt.title('GRL Agent')
# plt.show()

axis4 = fig.add_subplot(224)
dataframe9 = dataframe9.ewm(alpha = 0.2).mean()
dataframe10 = dataframe10.ewm(alpha = 0.2).mean()
sns.lineplot(x="Episode", y="Value", data=dataframe9,label='rl agent')
sns.lineplot(x="Episode", y="Value", data=dataframe10,label='grl agent')
# plt.plot(x="Step", y="Value", hue="Value", data=dataframe)
plt.xlabel('Episodes')
plt.ylabel('Step Time')
plt.legend()
# plt.title('GRL Agent')
# plt.show()
fig.tight_layout(pad = 1.0)
plt.savefig('training_results_smoothed.png',dpi=500)

fig = plt.figure(figsize=(10,8))
# dataframe7 = dataframe7.ewm(alpha = 0.2).mean()
# dataframe8 = dataframe8.ewm(alpha = 0.2).mean()
sns.lineplot(x="Episode", y="Value", data=dataframe7,label='rl agent')
sns.lineplot(x="Episode", y="Value", data=dataframe8,label='grl agent')
plt.xlabel('Episodes')
plt.ylabel('Reward Penalty')
plt.legend()
plt.savefig('training_results_smoothed2.png',dpi=500)