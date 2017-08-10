from __future__ import division

from collections import defaultdict
import pprint
import operator


def normalize_dictionary(dictionary, total):

    for key in dictionary:
        dictionary[key] /= total

    return dictionary


def get_label_stats(label_stats_file):

    label_stats = defaultdict(int)
    total_labels = 0

    with open(label_stats_file, "r") as label_stats_file:
        for stats_line in label_stats_file:
            if len(stats_line.split()) > 1:
                total_labels += 1
                label_stats[stats_line.split()[-1]] += 1

    return sorted(normalize_dictionary(dictionary=label_stats, total=total_labels).items(), key=operator.itemgetter(1))

    pass


def get_page_stats(page_stats_file):

    page_stats = defaultdict(int)
    total_docs = 0

    with open(page_stats_file, "r") as page_stats_file:
        for stats_line in page_stats_file:
            total_docs += 1
            for page_num in [int(page) for page in stats_line.split()[1:]]:
                page_stats[page_num] += 1

    return sorted(normalize_dictionary(dictionary=page_stats, total=total_docs).items(), key=operator.itemgetter(1))

    pass


def main():

    page_stats_file = "../../data/arxiv_page_stats.txt"
    label_stats_file = "../../data/arxiv_unordered.txt"
    pretty_print = pprint.PrettyPrinter(indent=4)

    print "Page Statistics: "
    pretty_print.pprint(get_page_stats(page_stats_file=page_stats_file))
    # print "Label Statistics: "
    # pretty_print.pprint(get_label_stats(label_stats_file=label_stats_file))

    pass


if __name__ == '__main__':
    main()
