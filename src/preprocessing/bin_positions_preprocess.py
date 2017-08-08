import numpy as np


def get_binned_positions(positions, num_x_bins, num_y_bins, page_coordinates):
    binned_positions = []

    x_bins = np.linspace(start=page_coordinates[0], stop=page_coordinates[2], num=num_x_bins)
    y_bins = np.linspace(start=page_coordinates[1], stop=page_coordinates[3], num=num_y_bins)

    for position in positions:
        binned_positions.append([str(np.digitize(position[0], x_bins)), str(np.digitize(position[1], y_bins)),
                                 str(np.digitize(position[2], x_bins)), str(np.digitize(position[3], y_bins))])

    return binned_positions


def bin_position_features(input_file, binned_file, num_x_bins, num_y_bins):
    tokens, labels, positions = [], [], []
    page_coordinates = [-1, -1, -1, -1]
    output_file = binned_file.split(".txt")[0] + "_" + str(num_x_bins) + "_" + str(num_y_bins) + ".txt"
    number_data_points = 0

    with open(output_file, "w") as binned_positions_file:
        with open(input_file, "r") as data_file:
            for data in data_file:
                if data != "\n":
                    if len(data.split()) > 1:
                        tokens.append(data.split()[0])
                        labels.append(data.split()[3])
                        positions.append([float(value) for value in data.split()[1].split(":")])
                    else:
                        page_coordinates = [int(coordinate) for coordinate in data.split(":")]
                else:
                    if len(tokens) and len(positions) and len(labels) and np.sum(page_coordinates) != -4:
                        number_data_points += 1
                        binned_positions = get_binned_positions(positions=np.array(positions), num_x_bins=num_x_bins,
                                                                num_y_bins=num_y_bins,
                                                                page_coordinates=page_coordinates)
                        for i in xrange(len(tokens)):
                            binned_positions_file.write(
                                " ".join([tokens[i], ":".join(binned_positions[i]), "*", labels[i]]) + "\n")
                        tokens, labels, positions = [], [], []
                        binned_positions_file.write("\n")

        if len(tokens) and len(positions) and len(labels) and np.sum(page_coordinates) != -4:
            number_data_points += 1
            binned_positions = get_binned_positions(positions=np.array(positions), num_x_bins=num_x_bins,
                                                    num_y_bins=num_y_bins,
                                                    page_coordinates=page_coordinates)
            for i in xrange(len(tokens)):
                binned_positions_file.write(
                    " ".join([tokens[i], ":".join(binned_positions[i]), "*", labels[i]]) + "\n")
            binned_positions_file.write("\n")

    return output_file, number_data_points


def train_test_split(input_file, output_dir, number_data_points):

    train_limit = int(0.6*number_data_points)
    dev_limit = train_limit + int(0.2*number_data_points)
    train_data_points, dev_data_points, test_data_points, data_point, i = [], [], [], [], 0

    # print train_limit, dev_limit

    with open(input_file, "r") as binned_positions_file:
        for data in binned_positions_file:
            if data == '\n' and len(data_point) > 0:
                if i < train_limit:
                    train_data_points.append(data_point)
                elif train_limit <= i < dev_limit:
                    dev_data_points.append(data_point)
                elif i >= dev_limit:
                    test_data_points.append(data_point)
                data_point = []
                i += 1
            elif data != '\n':
                data_point.append(data)

    with open(output_dir + "data.train", "w") as train_file:
        for data_point in train_data_points:
            for data in data_point:
                train_file.write(data)
            train_file.write("\n")

    with open(output_dir + "data.testa", "w") as dev_file:
        for data_point in dev_data_points:
            for data in data_point:
                dev_file.write(data)
            dev_file.write("\n")

    with open(output_dir + "data.testb", "w") as test_file:
        for data_point in test_data_points:
            for data in data_point:
                test_file.write(data)
            test_file.write("\n")


def main():

    num_x_bins, num_y_bins = 5, 5
    input_file, binned_file = "../../data/arxiv_unordered.txt", "../../data/arxiv_binned.txt"
    train_test_dir = "../../data/"

    output_file, number_data_points = bin_position_features(input_file=input_file, binned_file=binned_file, num_x_bins=num_x_bins, num_y_bins=num_y_bins)
    # print number_data_points
    print "Binned positions written into file: " + output_file

    train_test_split(input_file=output_file, output_dir=train_test_dir, number_data_points=number_data_points)
    print "Train, Dev, Test files written into respective files in directory: " + train_test_dir


if __name__ == '__main__':
    main()
