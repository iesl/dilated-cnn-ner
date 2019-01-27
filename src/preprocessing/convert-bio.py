from __future__ import print_function
import argparse

arg_parser = argparse.ArgumentParser(description='Convert I- labels to BIO')
arg_parser.add_argument('--input_file', type=str, help='Path to input file')
arg_parser.add_argument('--output_file', type=str, help='File to write')
args = arg_parser.parse_args()

FIELD_SEP = ' '

with open(args.input_file) as in_file, open(args.output_file, 'w') as output_file:
    last_label = 'Other'
    for line in in_file:
        line = line.strip()
        if line:
            split_line = line.split()
            full_label = split_line[3]
            rest = split_line[:3]
            label = full_label[2:]
            if label == "Other":
                new_label = "O"
            elif label == last_label:
                new_label = "I-" + label
            else:
                new_label = "B-" + label
            last_label = label
            fields = rest + [new_label]
            print(FIELD_SEP.join(fields), file=output_file)

        else:
            # # end of document
            # if buf:
            #     for tok in buf:
            #         print(tok, file=output_file)
            print(file=output_file)
            last_label = 'Other'

    # EOF; make sure we clear out the buffer
    # if buf:
    #     for tok_fields in buf:
    #         print(FIELD_SEP.join(tok_fields), file=output_file)