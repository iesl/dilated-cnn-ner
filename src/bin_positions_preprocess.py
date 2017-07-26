import numpy as np


def get_binned_positions(positions):
    binned_positions = []

    # print positions.shape
    min = np.min(a=positions, axis=0)[:2]
    max = np.max(a=positions, axis=0)[2:]
    x_bins = np.linspace(start=min[0], stop=max[0], num=4)
    y_bins = np.linspace(start=min[1], stop=max[1], num=4)

    for position in positions:
        binned_positions.append([str(np.digitize(position[0], x_bins)), str(np.digitize(position[1], y_bins)),
                                 str(np.digitize(position[2], x_bins)), str(np.digitize(position[3], y_bins))])

    return binned_positions


def main():
    tokens, labels, positions = [], [], []

    with open("../data/arxiv_binned.txt", "w") as binned_positions_file:
        with open("../data/arxiv_input_2.txt", "r") as data_file:
            for data in data_file:
                if data != "\n":
                    tokens.append(data.split()[0])
                    labels.append(data.split()[3])
                    positions.append([float(value) for value in data.split()[1].split(":")])
                else:
                    if len(tokens) and len(positions) and len(labels):
                        binned_positions = get_binned_positions(np.array(positions))
                        for i in xrange(len(tokens)):
                            binned_positions_file.write(
                                " ".join([tokens[i], ":".join(binned_positions[i]), "*", labels[i]]) + "\n")
                        tokens, labels, positions = [], [], []
                        binned_positions_file.write("\n")


if __name__ == '__main__':
    main()
