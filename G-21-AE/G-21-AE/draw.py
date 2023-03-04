import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator


def draw_reward_variation(episodes, rewards):
    plt.plot(episodes, rewards, linewidth=2)

    # 设置图片标题，并给坐标轴x,y分别加上标签
    plt.title('Chart of Reward Variation with Rounds', fontsize=15)
    plt.xlabel('Episodes', fontsize=12)
    plt.ylabel('Reward', fontsize=12)

    # 设置刻度标记的大小,axis='both'表示两坐标轴都设置
    plt.tick_params(axis='both', labelsize=10)
    plt.savefig('Chart of Reward Variation with Rounds.png')
    plt.show()
    plt.close()


def draw_mean_reward_variation(episodes, rewards):
    plt.plot(episodes, rewards, linewidth=2)
    x = MultipleLocator(10)
    y = MultipleLocator(20)
    # 设置图片标题，并给坐标轴x,y分别加上标签
    plt.title('Chart of Mean Reward Variation with Rounds', fontsize=15)
    plt.xlabel('Episodes', fontsize=12)
    plt.ylabel('Reward', fontsize=12)

    # 设置刻度标记的大小,axis='both'表示两坐标轴都设置
    plt.tick_params(axis='both', labelsize=10)
    ax = plt.gca()
    ax.xaxis.set_major_locator(x)
    ax.yaxis.set_major_locator(y)
    plt.savefig('Chart of Mean Reward Variation with Rounds.png')
    plt.show()
    plt.close()



