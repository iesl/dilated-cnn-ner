from __future__ import print_function
import argparse

arg_parser = argparse.ArgumentParser(description='Print various data statistics')
arg_parser.add_argument('--input_file', type=str, help='Path to input file')
args = arg_parser.parse_args()


num_documents = 0
label_counts = {}

with open(args.input_file) as in_file:
    buf = []
    for line in in_file:
        line = line.strip()
        if line:
            split_line = line.split()
            full_label = split_line[3]
            rest = split_line[:3]
            label = "O" if full_label == "O" else full_label[2:]
            bio_prefix = "O" if full_label == "O" else full_label[0]
            if bio_prefix == "B" or bio_prefix == "U":
                if label not in label_counts:
                    label_counts[label] = 0
                label_counts[label] += 1
            buf.append(line)

        elif buf:
            buf = []
            num_documents += 1
if buf:
    num_documents += 1


print("Number of documents: %d" % num_documents)
print("Number of phrases: %d" % sum(label_counts.values()))
print("Labeled segments:")
print(list(label_counts))
