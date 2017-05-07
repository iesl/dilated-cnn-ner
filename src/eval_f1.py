from __future__ import division
from __future__ import print_function
import time
import numpy as np
import sys

# compute various F1 statistics given a numpy array confusion matrix, and the index of the "outside" class
# if outside_idx=-1, there is no outside class
def compute_f1(confusion, label_map, gold_ax, outside_idx=-1):
    pred_ax = 1 if gold_ax == 0 else 0
    o_mask = np.zeros(len(label_map))
    if outside_idx > -1:
        o_mask[outside_idx] = 1

    tps = np.diag(confusion)
    tpfps = np.sum(confusion, axis=int(not pred_ax))
    tpfns = np.sum(confusion, axis=int(not gold_ax))

    masked_tps = np.ma.masked_array(tps, o_mask)
    masked_tpfps = np.ma.masked_array(tpfps, o_mask)
    masked_tpfns = np.ma.masked_array(tpfns, o_mask)

    all_correct = np.sum(tps)
    total = np.sum(confusion)
    accuracy = all_correct/total

    precisions = tps/tpfps
    recalls = tps/tpfns
    f1s = 2*precisions*recalls/(precisions+recalls)

    masked_precisions = np.ma.masked_array(precisions, o_mask)
    masked_recalls = np.ma.masked_array(recalls, o_mask)

    precision_macro = np.ma.mean(masked_precisions)
    recall_macro = np.ma.mean(masked_recalls)
    f1_macro = 2*precision_macro*recall_macro/(precision_macro+recall_macro)

    masked_tps_total = np.ma.sum(masked_tps)
    precision_micro = masked_tps_total/np.ma.sum(masked_tpfps)
    recall_micro = masked_tps_total/np.ma.sum(masked_tpfns)
    f1_micro = 2*precision_micro*recall_micro/(precision_micro+recall_micro)

    print("\t%10s\tPrec\tRecall\tAccuracy" % ("F1"))
    print("%10s\t%2.2f\t%2.2f\t%2.2f\t%2.2f" % ("Micro (Tok)", f1_micro*100, precision_micro*100, recall_micro*100, accuracy*100))
    print("%10s\t%2.2f\t%2.2f\t%2.2f" % ("Macro (Tok)", f1_macro*100, precision_macro*100, recall_macro*100))
    print("----------")
    for label in label_map:
        idx = label_map[label]
        if not idx == outside_idx:
            print("%10s\t%2.2f\t%2.2f\t%2.2f" % (label, f1s[idx]*100, precisions[idx]*100, recalls[idx]*100))
    sys.stdout.flush()


def token_eval(batches, predictions, label_map, type_int_int_map, outside_idx, pad_width, extra_text=""):
    num_types = len(label_map)
    confusion = np.zeros((num_types, num_types))
    if extra_text != "":
        print(extra_text)
    for predictions, (label_batch, token_batch, shape_batch, char_batch, seq_len_batch, mask_batch) in zip(predictions, batches):
        for preds, labels, seq_lens in zip(predictions, label_batch, seq_len_batch):
            start = pad_width
            for seq_len in seq_lens:
                for i in range(seq_len):
                # for pred, label in zip(preds[pad_width:seq_len+pad_width], labels[pad_width:seq_len+pad_width]):
                    # this will give you token-level F1
                    confusion[type_int_int_map[preds[i+start]], type_int_int_map[labels[i+start]]] += 1
                start += 2*pad_width + seq_len
    compute_f1(confusion, label_map, gold_ax=0, outside_idx=outside_idx)


def is_start(curr):
    return curr[0] == "B" or curr[0] == "U"


def is_continue(curr):
    return curr[0] == "I" or curr[0] == "L"


def is_background(curr):
    return not is_start(curr) and not is_continue(curr)


def is_seg_start(curr, prev):
    return (is_start(curr) and not is_continue(curr)) or (is_continue(curr) and (prev is None or is_background(prev) or prev[1:] != curr[1:]))

def segment_eval(batches, predictions, label_map, type_int_int_map, labels_id_str_map, vocab_id_str_map, outside_idx, pad_width, start_end, extra_text="", verbose=False):
    if extra_text != "":
        print(extra_text)

    def print_context(width, start, tok_list, pred_list, gold_list):
        for offset in range(-width, width+1):
            idx = offset + start
            if 0 <= idx < len(tok_list):
                print("%s\t%s\t%s" % (vocab_id_str_map[tok_list[idx]], labels_id_str_map[pred_list[idx]], labels_id_str_map[gold_list[idx]]))
        print()

    pred_counts = {t: 0 for t in label_map.values()}
    gold_counts = {t: 0 for t in label_map.values()}
    correct_counts = {t: 0 for t in label_map.values()}
    token_count = 0
    boundary_viols = 0
    type_viols = 0
    # iterate over batches
    for predictions, (dev_label_batch, dev_token_batch, dev_shape_batch, dev_char_batch, dev_seq_len_batch, dev_tok_len_batch, mask_batch) in zip(predictions, batches):
        # print(mask_batch[0])
        # iterate over examples in batch
        for preds, labels, tokens, seq_lens in zip(predictions, dev_label_batch, dev_token_batch, dev_seq_len_batch):
            start = pad_width
            # print(seq_lens)
            # print(start)
            # print(len(seq_lens))
            # print(len(preds))
            # print("sum seq lens", sum(seq_lens))
            # print("sum seq lens + pad", sum(seq_lens) + 2*(seq_lens != 0).sum()*pad_width)
            # print(preds)
            # print(labels)
            for seq_len in seq_lens:
                predicted = preds[start:seq_len+start]
                golds = labels[start:seq_len+start]
                toks = tokens[start:seq_len+start]
                # print("start: ", start)
                # print("len predicted: ", len(predicted))
                # print("predicted:")
                # print(predicted)
                # print(map(lambda p: labels_id_str_map[p], predicted))
                # print("gold:")
                # print(golds)
                # print(map(lambda p: labels_id_str_map[p], golds))
                for i in range(seq_len):
                    token_count += 1
                    pred = predicted[i]
                    gold = golds[i]
                    gold_str = labels_id_str_map[gold]
                    pred_str = labels_id_str_map[pred]
                    gold_prev = None if i == 0 else labels_id_str_map[golds[i - 1]]
                    pred_prev = None if i == 0 else labels_id_str_map[predicted[i - 1]]
                    pred_type = type_int_int_map[pred]
                    gold_type = type_int_int_map[gold]
                    pred_start = False
                    gold_start = False
                    if is_seg_start(pred_str, pred_prev):
                        pred_counts[pred_type] += 1
                        pred_start = True
                    if is_seg_start(gold_str, gold_prev):
                        gold_counts[gold_type] += 1
                        gold_start = True

                    if pred_start and gold_start:
                        # check for type violation
                        if pred_type != gold_type:
                            j = i + 1
                            stop_search = False
                            while j < seq_len and not stop_search:
                                pred2 = labels_id_str_map[predicted[j]]
                                gold2 = labels_id_str_map[golds[j]]
                                pred_type2 = type_int_int_map[predicted[j]]
                                pred_continue = is_continue(pred2)
                                gold_continue = is_continue(gold2)

                                if not pred_continue or not gold_continue or pred_type2 != gold_type or j == seq_len - 1:
                                    # check for type violation
                                    if pred_continue and gold_continue and pred_type2 != pred_type:
                                        type_viols += 1
                                        if verbose:
                                            print_context(2, j, toks, predicted, golds)
                                        stop_search = True
                                j += 1
                            # type_viols += 1
                            # if verbose:
                            #     print_context(2, i, toks, predicted, golds)
                        else:
                            if i == seq_len - 1:
                                correct_counts[gold_type] += 1
                            else:
                                j = i + 1
                                stop_search = False
                                while j < seq_len and not stop_search:
                                    pred2 = labels_id_str_map[predicted[j]]
                                    gold2 = labels_id_str_map[golds[j]]
                                    pred_type2 = type_int_int_map[predicted[j]]
                                    pred_continue = is_continue(pred2)
                                    gold_continue = is_continue(gold2)

                                    if not pred_continue or not gold_continue or pred_type2 != gold_type or j == seq_len - 1:
                                        # check for type violation
                                        if pred_continue and gold_continue and pred_type2 != pred_type:
                                            type_viols += 1
                                            if verbose:
                                                print_context(2, j, toks, predicted, golds)
                                        # check for boundary violation
                                        if pred_continue != gold_continue:
                                            # I or L must come after B or I
                                            if pred_continue:
                                                last_bilou = labels_id_str_map[predicted[j-1]][0]
                                                if last_bilou != "B" and last_bilou != "I":
                                                    boundary_viols += 1
                                                    if verbose:
                                                        print_context(2, j, toks, predicted, golds)
                                            if pred2[0] == "B":
                                                next_bilou = labels_id_str_map[predicted[j+1]][0] if j+1 < seq_len else "I" # wrong
                                                if next_bilou != "I" and next_bilou != "L":
                                                    boundary_viols += 1
                                                    if verbose:
                                                        print_context(2, j, toks, predicted, golds)

                                        # if pred_continue == gold_continue:
                                        if (not pred_continue and not gold_continue) or (pred_continue and gold_continue and pred_type2 == gold_type):
                                            correct_counts[gold_type] += 1
                                        stop_search = True
                                    j += 1
                start += seq_len + (2 if start_end else 1)*pad_width

    all_correct = np.sum([p if i not in outside_idx else 0 for i, p in enumerate(correct_counts.values())])
    all_pred = np.sum([p if i not in outside_idx else 0 for i, p in enumerate(pred_counts.values())])
    all_gold = np.sum([p if i not in outside_idx else 0 for i, p in enumerate(gold_counts.values())])

    precisions = [correct_counts[i] / pred_counts[i] if pred_counts[i] != 0 else 0.0 for i in pred_counts.keys()]
    recalls = [correct_counts[i] / gold_counts[i] if gold_counts[i] != 0 else 1.0 for i in gold_counts.keys()]
    f1s = [2 * precision * recall / (recall + precision) if recall + precision != 0 else 0.0 for precision, recall in
           zip(precisions, recalls)]

    precision_macro = np.mean(precisions)
    recall_macro = np.mean(recalls)
    f1_macro = 2 * precision_macro * recall_macro / (precision_macro + recall_macro)

    precision_micro = all_correct / all_pred
    recall_micro = all_correct / all_gold
    f1_micro = 2 * precision_micro * recall_micro / (precision_micro + recall_micro)

    accuracy = all_correct / all_gold

    print("\t%10s\tPrec\tRecall" % ("F1"))
    print("%10s\t%2.2f\t%2.2f\t%2.2f" % ("Micro (Seg)", f1_micro * 100, precision_micro * 100, recall_micro * 100))
    print("%10s\t%2.2f\t%2.2f\t%2.2f" % ("Macro (Seg)", f1_macro * 100, precision_macro * 100, recall_macro * 100))
    print("-------")
    for t in label_map:
        idx = label_map[t]
        if idx not in outside_idx:
            print("%10s\t%2.2f\t%2.2f\t%2.2f" % (t, f1s[idx] * 100, precisions[idx] * 100, recalls[idx] * 100))
    print("Processed %d tokens with %d phrases; found: %d phrases; correct: %d." % (token_count, all_gold, all_pred, all_correct))
    print("Found %d type violations, %d boundary violations." % (type_viols, boundary_viols))
    sys.stdout.flush()
    return f1_micro, precision_micro


def print_training_error(num_examples, start_time, epoch_losses, step):
    losses_str = ' '.join(["%5.5f"]*len(epoch_losses)) % tuple(map(lambda l: l/step, epoch_losses))
    print("%20d examples at %5.2f examples/sec. Error: %s" %
          (num_examples, num_examples / (time.time() - start_time), losses_str))
    sys.stdout.flush()

# label_map = {'LOC': 1, 'MISC': 4, 'O': 3, 'PER': 0, 'ORG': 2}
# confusion = np.array(
#     [[508, 29, 66, 53, 75],
#      [227, 1308, 227, 149, 104],
#      [143, 125, 508, 58, 59],
#      [2228, 592, 1225, 41765, 671],
#     [8, 14, 35, 12,  334]]
# )
# compute_f1(confusion, label_map, gold_ax=0, outside_idx=label_map["O"])


def print_conlleval_format(out_filename, eval_batches, predictions, labels_id_str_map, vocab_id_str_map, pad_width):
    with open(out_filename, 'w') as conll_preds_file:
        token_count = 0
        sentence_count = 0
        for prediction, (
        label_batch, token_batch, shape_batch, char_batch, seq_len_batch, tok_len_batch, eval_mask_batch) in zip(
                predictions, eval_batches):
            for preds, labels, tokens, seq_lens in zip(prediction, label_batch, token_batch, seq_len_batch):
                start = pad_width
                for seq_len in seq_lens:
                    if seq_len != 0:
                        preds_nopad = map(lambda t: labels_id_str_map[t], preds[start:seq_len + start])
                        labels_nopad = map(lambda t: labels_id_str_map[t], labels[start:seq_len + start])
                        tokens_nopad = map(lambda t: vocab_id_str_map[t], tokens[start:seq_len + start])
                        start += pad_width + seq_len
                        labels_converted = []
                        preds_converted = []
                        for idx, (pred, label) in enumerate(zip(preds_nopad, labels_nopad)):
                            token_count += 1

                            if pred[0] == "L":
                                preds_converted.append("I" + pred[1:])
                            elif pred[0] == "U":
                                preds_converted.append("B" + pred[1:])
                            else:
                                preds_converted.append(pred)

                            if label[0] == "L":
                                labels_converted.append("I" + label[1:])
                            elif label[0] == "U":
                                labels_converted.append("B" + label[1:])
                            else:
                                labels_converted.append(label)

                                # # O -> O
                                # if pred[0] == "O":
                                #     preds_converted.append(pred)
                                # # I,L -> I unless the last one was U or L converted to an I, and is the same type
                                # elif pred[0] == "I" or pred[0] == "L":
                                #     # this diff removed 1 correct and 5 found
                                #     # if preds_converted and (preds_nopad[idx-1][0] == "U" or preds_nopad[idx-1][0] == "L") and preds_converted[-1][1:] == pred[1:]:
                                #     if preds_converted and (preds_nopad[idx-1][0] == "L") and preds_converted[-1][1:] == pred[1:]:
                                #         preds_converted.append("B" + pred[1:])
                                #     else:
                                #         preds_converted.append("I" + pred[1:])
                                # # B,U -> I unless we are adjacent to an entity of the same type
                                # elif pred[0] == "B" or pred[0] == "U":
                                #     # this change removed 0 correct and 30 found
                                #     # if preds_converted and preds_converted[-1] != "O" and preds_converted[-1][1:] == pred[1:]:
                                #     if preds_converted and (preds_nopad[idx-1][0] == "L" or (preds_nopad[idx-1][0] == "I" and pred[0] == "B")) and preds_converted[-1][1:] == pred[1:]:
                                #         preds_converted.append("B" + pred[1:])
                                #     else:
                                #         preds_converted.append("I" + pred[1:])
                                #
                                # # O -> O
                                # if label[0] == "O":
                                #     labels_converted.append(label)
                                # # I,L -> I unless the last one was U or L converted to an I, and is the same type
                                # elif label[0] == "I" or label[0] == "L":
                                #     if labels_converted and (labels_nopad[idx-1][0] == "U" or labels_nopad[idx-1][0] == "L") and labels_converted[-1][1:] == label[1:]:
                                #         labels_converted.append("B" + label[1:])
                                #     else:
                                #         labels_converted.append("I" + label[1:])
                                # # B,U -> I unless we are adjacent to an entity of the same type
                                # elif label[0] == "B" or label[0] == "U":
                                #     if labels_converted and labels_converted[-1] != "O" and labels_converted[-1][1:] == label[1:]:
                                #         labels_converted.append("B" + label[1:])
                                #     else:
                                #         labels_converted.append("I" + label[1:])

                        for pred_conv, label_conv, pred, label, token in zip(preds_converted, labels_converted, preds_nopad,
                                                                             labels_nopad, tokens_nopad):
                            print("%s %s %s %s %s" % (token, label, pred, label_conv, pred_conv), file=conll_preds_file)
                        print("", file=conll_preds_file)
                        sentence_count += 1
                # print("%d tokens; %d sentences" % (token_count, sentence_count))