import pandas as pd
import numpy
import seaborn as sns
from matplotlib.ticker import FormatStrFormatter
import matplotlib.pyplot as plt
from scipy import stats


# grl_file = "run-ATC_GRL_Model_PPO_54-tag-rollout_ep_rew_mean.csv"
# rl_file = "run-ATC_RL_Model_PPO_8-tag-rollout_ep_rew_mean.csv"

grl_file = "run-ATC_GRL_Model_PPO_54-tag-ep_mean_collisions.csv"
rl_file = "run-ATC_RL_Model_PPO_8-tag-ep_mean_collisions.csv"

grl_df = pd.read_csv(grl_file, converters={"Step":int, "Value":float})
rl_df = pd.read_csv(rl_file, converters={"Step":int, "Value":float})
grl_df["Value_smoothed"] = grl_df["Value"].ewm(alpha=0.4).mean()

grl_df["Episodes"] = grl_df["Step"] / 1440
rl_df["Episodes"] = rl_df["Step"] / 1440


plt.figure()
sns.lineplot(data=grl_df, x="Episodes", y="Value")
sns.lineplot(data=rl_df, x="Episodes", y="Value")

plt.legend(labels=["grl agent", "rl agent"], title="Placeholder")
plt.title("Placeholder Collisions")
plt.xlabel("Episodes")
plt.ylabel("Collisions")
# plt.show()
plt.savefig("collision.png", dpi=300)


# grl_file = "run-ATC_GRL_Model_PPO_54-tag-ep_mean_battery.csv"
# rl_file = "run-ATC_RL_Model_PPO_8-tag-ep_mean_battery.csv"

# grl_df = pd.read_csv(grl_file, converters={"Step":int, "Value":float})
# rl_df = pd.read_csv(rl_file, converters={"Step":int, "Value":float})
# grl_df["Value_smoothed"] = grl_df["Value"].ewm(alpha=0.4).mean()

# grl_df["Episodes"] = (grl_df["Step"] / 1440).round(decimals=2)
# rl_df["Episodes"] = (rl_df["Step"] / 1440).round(decimals=2)

# barplot = pd.DataFrame({"Episodes":grl_df.Episodes, "grl":grl_df.Value, "rl":rl_df.Value})

# fig, axes = plt.subplots(2,2)
# # sns.lineplot(data=grl_df, x="Episodes", y="Value", ax=axes[0, 0])
# # sns.lineplot(data=rl_df, x="Episodes", y="Value", ax=axes[0, 0])

# barplot[::10].plot(kind="bar", stacked="True", x="Episodes", ax=axes[0, 1])

# barplot.plot(x="Episodes", ax=axes[0, 0])

# barplot.plot(x="Episodes", kind="box", ax=axes[1, 0])

# barplot.plot(x="Episodes", kind="box", ax=axes[1, 1], vert=False)

# axes[0, 1].xaxis.set_major_formatter(FormatStrFormatter('%.2f'))


# plt.legend(labels=["grl agent", "rl agent"], title="Placeholder")
# plt.title("Placeholder reward")
# plt.xlabel("Episodes")
# plt.ylabel("Reward")
# fig.tight_layout()
# plt.show()
# plt.savefig("reward.png", dpi=300)