from __future__ import division

from collections import defaultdict
import pprint
import operator
import numpy as np


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


def compute_page_stats(page_stats_fp, thresholds=(0.1, 0.2)):

    docs_within_thresholds = defaultdict(int)
    page_numbers_dict = defaultdict(int)
    total_docs = 0
    pretty_print = pprint.PrettyPrinter(indent=4)

    threshold_violations_file = "../../data/arxiv_metadata_on_multiple_pages.txt"
    threshold_violations = {}
    total_threshold_violations = 0

    with open(page_stats_fp, "r") as page_stats_fp:
        for stats_line in page_stats_fp:

            page_numbers = np.array([int(page_number) for page_number in stats_line.split()[1:-1]])
            total_pages = int(stats_line.split()[-1])
            metadata_page_boundary = 0
            if np.sum(page_numbers) > 1:
                metadata_page_boundary = np.max(page_numbers)/total_pages
            for threshold in thresholds:
                if metadata_page_boundary <= threshold:
                    docs_within_thresholds[threshold] += 1
                else:
                    if threshold not in threshold_violations:
                        threshold_violations[threshold] = []
                    threshold_violations[threshold].append(stats_line.split()[0])
                    total_threshold_violations += 1

            total_docs += 1
            for page_number in page_numbers:
                page_numbers_dict[page_number] += 1

    if total_threshold_violations > 0:
        with open(threshold_violations_file, "w") as threshold_violations_fp:
            threshold_violations_fp.write("Threshold\tDoc Id\t\tDoc Stable Id \n")
            for threshold in threshold_violations:
                for threshold_violation in threshold_violations[threshold]:
                    threshold_violations_fp.write(str(threshold) + "\t\t\t" + "\t\t\t".join(threshold_violation.split(":")) + "\n")

    print "Page Statistics: "

    print "\nPage numbers: (page_number, frequency)"
    pretty_print.pprint(sorted(normalize_dictionary(dictionary=page_numbers_dict, total=total_docs).items(), key=operator.itemgetter(1), reverse=True))

    print "\nPapers with Metadata pages within thresholds: (threshold, frequency)"
    pretty_print.pprint(sorted(normalize_dictionary(dictionary=docs_within_thresholds, total=total_docs).items(), key=operator.itemgetter(0)))

    print "\n{} documents with respective threshold violations are written to file {}".format(total_threshold_violations, threshold_violations_file)

    pass


def main():

    page_stats_file = "../../data/arxiv_page_stats.txt"
    label_stats_file = "../../data/arxiv_unordered.txt"

    pretty_print = pprint.PrettyPrinter(indent=4)

    compute_page_stats(page_stats_fp=page_stats_file, thresholds=(0.1, 0.2))

    # print "Label Statistics: "
    # pretty_print.pprint(get_label_stats(label_stats_file=label_stats_file))

    pass


if __name__ == '__main__':
    main()
