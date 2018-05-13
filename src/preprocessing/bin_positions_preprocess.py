from __future__ import division
import numpy as np
import math
import sys, getopt


def get_binned_height_width(positions, num_height_bins, num_width_bins, height_indices, width_indices):
    height_bins = np.linspace(start=positions.min(axis=0)[3], stop=positions.max(axis=0)[3], num=num_height_bins)
    width_bins = np.linspace(start=positions.min(axis=0)[2], stop=positions.max(axis=0)[2], num=num_width_bins)
    binned_positions = []

    for i, position in enumerate(positions):
        binned_measures = []
        for index, measurement in enumerate(position):
            if index in height_indices:
                binned_measures.append(str(np.digitize(measurement, height_bins)))
            elif index in width_indices:
                binned_measures.append(str(np.digitize(measurement, width_bins)))
            else:
                binned_measures.append(str(measurement))
        binned_positions.append(binned_measures)

    return binned_positions


def get_centroid_measurements(position):
    # page_number = int(position[0])
    centroid_x = int(position[1]) + ((int(position[3]) - int(position[1])) / 2)
    centroid_y = int(position[2]) + ((int(position[4]) - int(position[2])) / 2)
    width = int(position[3]) - int(position[1])
    height = int(position[4]) - int(position[2])
    # width_height_ratio = (int(position[3]) - int(position[1])) / (int(position[4]) - int(position[2]))

    return [centroid_x, centroid_y, width, height]


def get_binned_positions(positions, num_x_bins, num_y_bins, page_coordinates, x_bin_indices, y_bin_indices):
    binned_positions = []

    y_bins = np.linspace(start=positions.min(axis=0)[1], stop=positions.max(axis=0)[1], num=num_y_bins)
    x_bins = np.linspace(start=positions.min(axis=0)[0], stop=positions.max(axis=0)[0], num=num_x_bins)

    # x_bins = np.linspace(start=page_coordinates[0], stop=page_coordinates[2], num=num_x_bins)
    # y_bins = np.linspace(start=page_coordinates[1], stop=page_coordinates[3], num=num_y_bins)

    for i, position in enumerate(positions):
        binned_position = []
        for index, local_position in enumerate(position):
            if index in x_bin_indices:
                binned_position.append(str(np.digitize(local_position, x_bins)))
            elif index in y_bin_indices:
                binned_position.append(str(np.digitize(local_position, y_bins)))
            else:
                binned_position.append(str(local_position))
        binned_positions.append(binned_position)

    return binned_positions


def bin_position_features(input_file, binned_file, num_x_bins, num_y_bins):
    tokens, labels, positions = [], [], []
    page_coordinates = [-1, -1, -1, -1]
    output_file = binned_file.split(".txt")[0] + "_" + str(num_x_bins) + "_" + str(num_y_bins) + ".txt"
    data_points_count = 0

    with open(output_file, "w") as binned_positions_file:
        with open(input_file, "r") as data_file:
            for data in data_file:
                if data != "\n":
                    if len(data.split()) > 1:
                        tokens.append(data.split()[0])
                        labels.append(data.split()[3])
                        positions.append([float(value) for value in data.split()[1].split(":")[1:]])
                    else:
                        page_coordinates = [int(coordinate) for coordinate in data.split(":")]
                else:
                    if len(tokens) and len(positions) and len(labels) and np.sum(page_coordinates) != -4:
                        data_points_count += 1
                        binned_positions = get_binned_positions(positions=np.array(positions), num_x_bins=num_x_bins,
                                                                num_y_bins=num_y_bins,
                                                                page_coordinates=page_coordinates, x_bin_indices=(0, 1),
                                                                y_bin_indices=(2, 3))
                        for i in xrange(len(tokens)):
                            binned_positions_file.write(
                                " ".join([tokens[i], ":".join(binned_positions[i]), "*", labels[i]]) + "\n")
                        tokens, labels, positions = [], [], []
                        binned_positions_file.write("\n")

        if len(tokens) and len(positions) and len(labels) and np.sum(page_coordinates) != -4:
            data_points_count += 1
            binned_positions = get_binned_positions(positions=np.array(positions), num_x_bins=num_x_bins,
                                                    num_y_bins=num_y_bins,
                                                    page_coordinates=page_coordinates, x_bin_indices=(0, 1),
                                                    y_bin_indices=(2, 3))
            for i in xrange(len(tokens)):
                binned_positions_file.write(
                    " ".join([tokens[i], ":".join(binned_positions[i]), "*", labels[i]]) + "\n")
            binned_positions_file.write("\n")

    return output_file, data_points_count


def train_test_split(input_file, output_dir, number_data_points):
    train_limit = int(math.ceil(0.6 * number_data_points))
    dev_limit = int(math.ceil(0.8 * number_data_points))
    train_data_points, dev_data_points, test_data_points, data_point, i = [], [], [], [], 0

    shuffled_indices = range(number_data_points)
    np.random.shuffle(shuffled_indices)

    train_indices = shuffled_indices[:train_limit]
    print len(train_indices)
    dev_indices = shuffled_indices[train_limit:dev_limit]
    print len(dev_indices)
    test_indices = shuffled_indices[dev_limit:]
    print len(test_indices)

    # print train_limit, dev_limit

    with open(input_file, "r") as binned_positions_file:
        for data in binned_positions_file:
            if data == '\n' and len(data_point) > 0:
                if i in train_indices:
                    train_data_points.append(data_point)
                elif i in dev_indices:
                    dev_data_points.append(data_point)
                elif i in test_indices:
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


def get_binned_centroid_measurements(input_file, binned_file, num_x_bins, num_y_bins):
    tokens, labels, positions = [], [], []
    page_coordinates = [-1, -1, -1, -1]
    data_points_count = 0
    output_file = binned_file.split(".txt")[0] + "_" + str(num_x_bins) + "_" + str(num_y_bins) + ".txt"

    with open(output_file, "w") as binned_positions_file:
        with open(input_file, "r") as data_file:
            for data in data_file:
                if data != "\n":
                    if len(data.split()) > 1:
                        tokens.append(data.split()[0])
                        labels.append(data.split()[3])
                        positions.append(
                            [float(value) for value in get_centroid_measurements(data.split()[1].split(":"))])
                    else:
                        page_coordinates = [int(coordinate) for coordinate in data.split(":")]
                else:
                    if len(tokens) and len(positions) and len(labels) and np.sum(page_coordinates) != -4:
                        data_points_count += 1
                        binned_positions = get_binned_height_width(positions=np.array(positions), num_height_bins=2,
                                                                   num_width_bins=2, height_indices=(3,),
                                                                   width_indices=(2,))
                        binned_positions = get_binned_positions(positions=np.array(binned_positions),
                                                                num_x_bins=num_x_bins,
                                                                num_y_bins=num_y_bins,
                                                                page_coordinates=page_coordinates, x_bin_indices=(0,),
                                                                y_bin_indices=(1,))
                        for i in xrange(len(tokens)):
                            binned_positions_file.write(
                                " ".join([tokens[i], ":".join(binned_positions[i]), "*", labels[i]]) + "\n")
                        tokens, labels, positions = [], [], []
                        binned_positions_file.write("\n")

        if len(tokens) and len(positions) and len(labels) and np.sum(page_coordinates) != -4:
            data_points_count += 1
            binned_positions = get_binned_height_width(positions=np.array(positions), num_height_bins=2,
                                                       num_width_bins=2, height_indices=(3,),
                                                       width_indices=(2,))
            binned_positions = get_binned_positions(positions=np.array(binned_positions), num_x_bins=num_x_bins,
                                                    num_y_bins=num_y_bins,
                                                    page_coordinates=page_coordinates, x_bin_indices=(0,),
                                                    y_bin_indices=(1,))
            for i in xrange(len(tokens)):
                binned_positions_file.write(
                    " ".join([tokens[i], ":".join(binned_positions[i]), "*", labels[i]]) + "\n")
            binned_positions_file.write("\n")

    return output_file, data_points_count


def main(argv):
    num_x_bins, num_y_bins = 0, 0
    input_file, binned_file, train_test_dir = "", "", ""

    try:
        opts, args = getopt.getopt(argv, "hx:y:i:b:d:", ["xbins=", "ybins=", "input=", "binned=", "traintestdir="])
    except getopt.GetoptError:
        print 'python bin_positions_preprocess.py -x <x_bins> -y <y_bing> -i <input_file> -b <binned_output_file> -d <train_test_dir>'
        sys.exit(2)

    for opt, arg in opts:
        if opt == '-h':
            print 'python bin_positions_preprocess.py -x <x_bins> -y <y_bing> -i <input_file> -b <binned_output_file> -d <train_test_dir>'
            sys.exit()
        elif opt in ("-x", "--xbins"):
            num_x_bins = arg
        elif opt in ("-y", "--ybins"):
            num_y_bins = arg
        elif opt in ("-i", "--input"):
            input_file = arg
        elif opt in ("-b", "--binned"):
            binned_file = arg
        elif opt in ("-d", "--traintestdir"):
            train_test_dir = arg

    output_file, data_points_count = get_binned_centroid_measurements(input_file=input_file, binned_file=binned_file,
                                                                      num_x_bins=num_x_bins, num_y_bins=num_y_bins)
    print "{} Binned positions written into file: {}".format(data_points_count, output_file)

    train_test_split(input_file=output_file, output_dir=train_test_dir, number_data_points=data_points_count)
    print "Train, Dev, Test files written into respective files in directory: " + train_test_dir


if __name__ == '__main__':
    main(sys.argv[1:])
