

import pandas as pd
import seaborn as sns

from matplotlib import pyplot as plt

# from pnplot import *

# dataset1_filename = 'run-ATC_GRL_Model_PPO_7-tag-rollout_ep_rew_mean.csv'
# dataset2_filename = 'run-ATC_GRL_Model_PPO_7-tag-train_entropy_loss.csv'
dataset1_filename = 'PPO3_Reward.csv'
dataset2_filename = 'PPO3_Entropy.csv'


dataframe1 = pd.read_csv(open(dataset1_filename, 'rb'))
dataframe2 = pd.read_csv(open(dataset2_filename, 'rb'))

fig = plt.figure(figsize=(10,8))
axis = fig.add_subplot(211)

#Mission Time Figure
sns.lineplot(x="Episode", y="Value", data=dataframe1,palette=['blue'])
# plt.plot(x="Step", y="Value", hue="Value", data=dataframe)
plt.xlabel('Episodes')
plt.ylabel('Episode Reward Mean')
# plt.title('GRL Agent')
# plt.show()

axis2 = fig.add_subplot(212)
sns.lineplot(x="Episode", y="Value", data=dataframe2,palette=['blue'])
# plt.plot(x="Step", y="Value", hue="Value", data=dataframe)
plt.xlabel('Episodes')
plt.ylabel('Episode Entropy Loss')
# plt.title('GRL Agent')
# plt.show()
plt.savefig('Posterplot.png',dpi=500)
