import matplotlib.pyplot as plt
import os

def read_file(filename):
    stats={}
    with open(filename) as file:
        for line in file:
            i = line.split()[0]

            value = float(line.split()[1].rstrip())
            stats[i]=value
    return stats


def create_graph(feature):
    params = {'legend.fontsize': 'x-large',
              'figure.figsize': (13, 8),
              'axes.labelsize': 'x-large',
              'axes.titlesize': 'x-large',
              'xtick.labelsize': 'x-large',
              'ytick.labelsize': 'x-large',
              'font.family': 'serif'}
    plt.rcParams.update(params)
    group_name_dict = {"bot": "Bot", "active": "Students", "static": "Static", "top": "S-T","dummy":"Planted"}
    colors_dict = {"bot": "b", "active": "r", "static": "y", "top": "k","dummy":"mediumslateblue"}
    axis_dict = {"average": "Average Rank", "raw": "Raw promotion", "potential": "Scaled promotion","ks":"Quality","rel":"Relevance"}
    dot_dict = {"bot": "-o", "active": "--^", "static": ":p", "top": "-.x","dummy":"-.+"}


    plt.figure()
    features_stats_dir = "stats"
    for file in os.listdir(features_stats_dir):
        if not file.__contains__(feature):
            continue
        filename = features_stats_dir + "/" + file
        group = file.split("_")[0]
        if group == "top":
            continue
        stats = read_file(filename)

        x = [j + 1 for j in range(len(stats)-1)]
        if feature in ["raw","potential"]:
            x = [j + 2 for j in range(len(stats)-1)]
        if "0" in stats or "8" in stats or "9" in stats:
            y = [stats[str(i)] for i in sorted([int(i) for i in list(stats.keys())])][:-1]
        else:
            y = [stats[str(i)] for i in sorted(list(stats.keys()))][:-1]

        plt.plot(x, y, dot_dict[group], label=group_name_dict[group], color=colors_dict[group], linewidth=5,
                 markersize=10, mew=1)
    # plt.xticks(x)
    plt.xticks(x, fontsize=25)
    plt.yticks(fontsize=25)
    plt.ylabel(axis_dict[feature], fontsize=30)
    # plt.ylabel(axis_dict[feature])
    plt.xlabel("Round", fontsize=30)
    # plt.grid(True)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.11),
               ncol=5, fontsize=25, frameon=False)
    plt.savefig(feature + ".pdf", format="pdf")
    # plt.savefig(feature)
    # plt.show()
    plt.clf()

create_graph("potential")
# create_graph("raw")
# create_graph("average")
# create_graph("rel")
# create_graph("ks")