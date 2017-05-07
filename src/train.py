from __future__ import division
from __future__ import print_function
import sys
import time
import tensorflow as tf
import numpy as np
from data_utils import SeqBatcher, Batcher
from cnn_spred import CNN_Spred
from cnn_spred_multitask import CNN_Spred_Multi
from cnn_fast import CNN_Fast
from bilstm import BiLSTM
from bilstm_char import BiLSTMChar
from cnn_char import CNNChar
from context_agg import ContextAgg
from encoder_decoder import EncoderDecoder
import eval_f1 as evaluation
import json
import tf_utils
from os import listdir
import os

FLAGS = tf.app.flags.FLAGS


def main(argv):
    print("CUDA_VISIBLE_DEVICES=", os.environ['CUDA_VISIBLE_DEVICES'])

    train_dir = FLAGS.train_dir
    dev_dir = FLAGS.dev_dir
    maps_dir = FLAGS.maps_dir

    if train_dir == '':
        print('Must supply input data directory generated from tsv_to_tfrecords.py')
        sys.exit(1)

    print('\n'.join(sorted(["%s : %s" % (str(k), str(v)) for k, v in FLAGS.__dict__['__flags'].items()])))

    with open(maps_dir + '/label.txt', 'r') as f:
        labels_str_id_map = {l.split('\t')[0]: int(l.split('\t')[1].strip()) for l in f.readlines()}
        labels_id_str_map = {i: s for s, i in labels_str_id_map.items()}
        labels_size = len(labels_id_str_map)
    with open(maps_dir + '/token.txt', 'r') as f:
        vocab_str_id_map = {l.split('\t')[0]: int(l.split('\t')[1].strip()) for l in f.readlines()}
        vocab_id_str_map = {i: s for s, i in vocab_str_id_map.items()}
        vocab_size = len(vocab_id_str_map)
    with open(maps_dir + '/shape.txt', 'r') as f:
        shape_str_id_map = {l.split('\t')[0]: int(l.split('\t')[1].strip()) for l in f.readlines()}
        shape_id_str_map = {i: s for s, i in shape_str_id_map.items()}
        shape_domain_size = len(shape_id_str_map)
    with open(maps_dir + '/char.txt', 'r') as f:
        char_str_id_map = {l.split('\t')[0]: int(l.split('\t')[1].strip()) for l in f.readlines()}
        char_id_str_map = {i: s for s, i in char_str_id_map.items()}
        char_domain_size = len(char_id_str_map)

    # with open(maps_dir + '/sizes.txt', 'r') as f:
    #     num_train_examples = int(f.readline()[:-1])

    print("num classes: %d" % labels_size)

    size_files = [maps_dir + "/" + fname for fname in listdir(maps_dir) if fname.find("sizes") != -1]
    num_train_examples = 0
    num_tokens = 0
    for size_file in size_files:
        print(size_file)
        with open(size_file, 'r') as f:
            num_train_examples += int(f.readline()[:-1])
            num_tokens += int(f.readline()[:-1])

    print("num train examples: %d" % num_train_examples)
    print("num train tokens: %d" % num_tokens)

    dev_top_dir = '/'.join(dev_dir.split("/")[:-2]) if dev_dir.find("*") != -1 else dev_dir
    print(dev_top_dir)
    dev_size_files = [dev_top_dir + "/" + fname for fname in listdir(dev_top_dir) if fname.find("sizes") != -1]
    num_dev_examples = 0
    num_dev_tokens = 0
    for size_file in dev_size_files:
        print(size_file)
        with open(size_file, 'r') as f:
            num_dev_examples += int(f.readline()[:-1])
            num_dev_tokens += int(f.readline()[:-1])

    print("num dev examples: %d" % num_dev_examples)
    print("num dev tokens: %d" % num_dev_tokens)

    # with open(dev_dir + '/sizes.txt', 'r') as f:
    #     num_dev_examples = int(f.readline()[:-1])

    type_int_int_map = {}
    bilou_int_int_map = {}
    bilou_set = {}
    type_set = {}
    outside_set = ["O", "<PAD>",  "<S>",  "</S>", "<ZERO>"]
    for label, id in labels_str_id_map.items():
        label_type = label if label in outside_set else label[2:]
        label_bilou = label[0]
        if label_type not in type_set:
            type_set[label_type] = len(type_set)
        if label_bilou not in bilou_set:
            bilou_set[label_bilou] = len(bilou_set)
        type_int_int_map[id] = type_set[label_type]
        bilou_int_int_map[id] = bilou_set[label_bilou]

    type_int_str_map = {a: b for b, a in type_set.items()}
    bilou_int_str_map = {a: b for b, a in bilou_set.items()}
    num_types = len(type_set)
    num_bilou = len(bilou_set)
    print(type_set)

    # load embeddings, if given; initialize in range [-.01, .01]
    embeddings_shape = (vocab_size-1, FLAGS.embed_dim)
    embeddings = tf_utils.embedding_values(embeddings_shape, old=False)
    embeddings_used = 0
    if FLAGS.embeddings != '':
        with open(FLAGS.embeddings, 'r') as f:
            for line in f.readlines():
                split_line = line.strip().split(" ")
                word = split_line[0]
                embedding = split_line[1:]
                # print("word: %s" % word)
                # print("embedding: %s" % embedding)
                if word in vocab_str_id_map and (FLAGS.pretrained_pad or (word != "<PAD>" and word != "<OOV>")):
                    embeddings_used += 1
                    # shift by -1 because we are going to add a 0 constant vector for the padding later
                    embeddings[vocab_str_id_map[word]-1] = map(float, embedding)
                elif word.lower() in vocab_str_id_map and (FLAGS.pretrained_pad or (word != "<PAD>" and word != "<OOV>")):
                    embeddings_used += 1
                    embeddings[vocab_str_id_map[word.lower()] - 1] = map(float, embedding)
    print("Loaded %d/%d embeddings (%2.2f%% coverage)" % (embeddings_used, vocab_size, embeddings_used/vocab_size*100))

    layers_map = sorted(json.loads(FLAGS.layers.replace("'", '"')).items()) \
                    if FLAGS.model == 'cnn' or FLAGS.model == "cnn-fast" or FLAGS.model == "cnn-multi" \
                    else None
    layers_map2 = sorted(json.loads(FLAGS.layers2.replace("'", '"')).items()) \
                    if layers_map is not None and FLAGS.layers2 != '' \
                    else None

    pad_width = int(layers_map[0][1]['width']/2) if layers_map is not None else 1

    # if FLAGS.model == 'cnn' or FLAGS.model == "cnn-fast":
    #     layers = FLAGS.layers.replace("'", '"')
    #     print("layers: ", layers)
    #     layers_map = sorted(json.loads(layers).items())
    #
    #     if FLAGS.layers2 != '':
    #         layers2 = FLAGS.layers2.replace("'", '"')
    #         print("layers2: ", layers2)
    #         layers_map2 = sorted(json.loads(layers2).items())
    #     input_filter_width = layers_map[0][1]['width']
    #     pad_width = int(input_filter_width/2)
    # else:
    #     pad_width = 1

    def sample_pad_size():
        return np.random.randint(1, FLAGS.max_additional_pad) if FLAGS.max_additional_pad > 0 else pad_width

    # print(seq_len_with_pad)

    with tf.Graph().as_default():
        # train_batcher = NodeBatcher(train_dir, seq_len_with_pad, FLAGS.batch_size)
        # num_buckets = 2 #int(num_train_examples/FLAGS.batch_size) + (0 if num_train_examples % FLAGS.batch_size == 0 else 1)
        # print("num buckets: %d" % num_buckets)
        train_batcher = Batcher(train_dir, FLAGS.batch_size) if FLAGS.memmap_train else SeqBatcher(train_dir, FLAGS.batch_size)

        dev_batch_size = FLAGS.batch_size # num_dev_examples
        # dev_batcher = NodeBatcher(dev_dir, seq_len_with_pad, dev_batch_size, num_epochs=1)
        dev_batcher = SeqBatcher(dev_dir, dev_batch_size, num_buckets=0, num_epochs=1)
        if FLAGS.ontonotes:
            domain_dev_batchers = {domain: SeqBatcher(dev_dir.replace('*', domain),
                                                      dev_batch_size, num_buckets=0, num_epochs=1)
                                   for domain in ['bc', 'nw', 'bn', 'wb', 'mz', 'tc']}

        train_eval_batch_size = FLAGS.batch_size #num_train_examples
        # train_eval_batcher = NodeBatcher(train_dir, seq_len_with_pad, train_eval_batch_size, num_epochs=1)
        train_eval_batcher = SeqBatcher(train_dir, train_eval_batch_size, num_buckets=0, num_epochs=1)

        char_embedding_model = BiLSTMChar(char_domain_size, FLAGS.char_dim, int(FLAGS.char_tok_dim/2)) \
            if FLAGS.char_dim > 0 and FLAGS.char_model == "lstm" else \
            (CNNChar(char_domain_size, FLAGS.char_dim, FLAGS.char_tok_dim, layers_map[0][1]['width'])
                if FLAGS.char_dim > 0 and FLAGS.char_model == "cnn" else None)
        char_embeddings = char_embedding_model.outputs if char_embedding_model is not None else None

        if FLAGS.model == 'cnn':
            model = CNN_Spred(
                    num_classes=labels_size,
                    vocab_size=vocab_size,
                    shape_domain_size=shape_domain_size,
                    char_domain_size=char_domain_size,
                    char_size=FLAGS.char_tok_dim,
                    embedding_size=FLAGS.embed_dim,
                    shape_size=FLAGS.shape_dim,
                    nonlinearity=FLAGS.nonlinearity,
                    layers_map=layers_map,
                    viterbi=FLAGS.viterbi,
                    res_activation=FLAGS.frontend_residual_layers,
                    batch_norm=FLAGS.frontend_batch_norm,
                    projection=FLAGS.projection,
                    loss=FLAGS.loss,
                    margin=FLAGS.margin,
                    repeats=FLAGS.block_repeats,
                    share_repeats=FLAGS.share_repeats,
                    char_embeddings=char_embeddings,
                    pool_blocks=FLAGS.pool_blocks,
                    fancy_blocks=FLAGS.fancy_blocks,
                    residual_blocks=FLAGS.residual_blocks,
                    embeddings=embeddings)
        elif FLAGS.model == "cnn-multi":
            model = CNN_Spred_Multi(
                num_label_classes=labels_size,
                num_bio_classes=num_bilou,
                num_type_classes=num_types,
                vocab_size=vocab_size,
                shape_domain_size=shape_domain_size,
                char_domain_size=char_domain_size,
                char_size=FLAGS.char_dim,
                embedding_size=FLAGS.embed_dim,
                shape_size=FLAGS.shape_dim,
                nonlinearity=FLAGS.nonlinearity,
                layers_map=layers_map,
                viterbi=FLAGS.viterbi,
                res_activation=FLAGS.frontend_residual_layers,
                batch_norm=FLAGS.frontend_batch_norm,
                projection=FLAGS.projection,
                loss=FLAGS.loss,
                embeddings=embeddings)
        elif FLAGS.model == "cnn-fast":
            model = CNN_Fast(
                # sequence_length=seq_len_with_pad,
                num_classes=labels_size,
                vocab_size=vocab_size,
                shape_domain_size=shape_domain_size,
                char_domain_size=char_domain_size,
                char_size=FLAGS.char_dim,
                embedding_size=FLAGS.embed_dim,
                shape_size=FLAGS.shape_dim,
                nonlinearity=FLAGS.nonlinearity,
                layers_map=layers_map,
                viterbi=FLAGS.viterbi,
                layers_map2=None if FLAGS.layers2 == '' else layers_map2,
                embeddings=embeddings)
        elif FLAGS.model == "bilstm":
            model = BiLSTM(
                    num_classes=labels_size,
                    vocab_size=vocab_size,
                    shape_domain_size=shape_domain_size,
                    char_domain_size=char_domain_size,
                    char_size=FLAGS.char_dim,
                    embedding_size=FLAGS.embed_dim,
                    shape_size=FLAGS.shape_dim,
                    nonlinearity=FLAGS.nonlinearity,
                    viterbi=FLAGS.viterbi,
                    hidden_dim=FLAGS.lstm_dim,
                    char_embeddings=char_embeddings,
                    embeddings=embeddings)
        elif FLAGS.model == 'seq2seq':
            model = EncoderDecoder(
                    # sequence_length=seq_len_with_pad,
                    num_classes=labels_size,
                    vocab_size=vocab_size,
                    embedding_size=FLAGS.embed_dim,
                    lstm_dim=FLAGS.lstm_dim)
        else:
            print(FLAGS.model + ' is not a valid model type')
            sys.exit(1)

        context_agg = None if FLAGS.layers2 == '' else \
                        ContextAgg(
                            num_classes=labels_size,
                            embedding_size=labels_size,
                            nonlinearity=FLAGS.nonlinearity,
                            layers_map=layers_map2,
                            viterbi=FLAGS.viterbi,
                            frontend_outputs=model.unflat_scores,
                            frontend_no_dropout=model.unflat_no_dropout_scores,
                            frontend_sample_pad=model.unflat_sample_pad_scores,
                            res_activation=FLAGS.context_residual_layers,
                            batch_norm=FLAGS.context_batch_norm,
                            loss=FLAGS.loss,
                            margin=FLAGS.margin
                        )

        # Define Training procedure
        global_step = tf.Variable(0, name='global_step', trainable=False)
        global_step_context = tf.Variable(0, name='context_agg_global_step', trainable=False)
        global_step_all = tf.Variable(0, name='context_agg_all_global_step', trainable=False)

        optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.lr, beta1=FLAGS.beta1, beta2=FLAGS.beta2, epsilon=FLAGS.epsilon, name="optimizer")

        model_vars = [v for v in tf.global_variables() if 'context_agg' not in v.name]
        context_agg_vars = [v for v in tf.global_variables() if 'context_agg' in v.name]

        print("model vars: %d" % len(model_vars))
        print(map(lambda v: v.name, model_vars))
        print("context vars: %d" % len(context_agg_vars))
        print(map(lambda v: v.name, context_agg_vars))

        # todo put in func
        total_parameters = 0
        for variable in tf.trainable_variables():
            # shape is an array of tf.Dimension
            shape = variable.get_shape()
            variable_parametes = 1
            for dim in shape:
                variable_parametes *= dim.value
            total_parameters += variable_parametes
        print("Total trainable parameters: %d" % (total_parameters))

        if FLAGS.clip_norm > 0:
            grads, _ = tf.clip_by_global_norm(tf.gradients(model.loss, model_vars), FLAGS.clip_norm)
            train_op = optimizer.apply_gradients(zip(grads, model_vars), global_step=global_step)
        else:
            train_op = optimizer.minimize(model.loss, global_step=global_step, var_list=model_vars)
        # grads_and_vars = optimizer.compute_gradients(model.loss)
        # train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

        if FLAGS.layers2 != '':
            context_agg_train_op = optimizer.minimize(context_agg.loss, global_step=global_step_context, var_list=context_agg_vars)
            all_train_op = optimizer.minimize(context_agg.loss, global_step=global_step_all)

        tf.global_variables_initializer()

        frontend_opt_vars = [optimizer.get_slot(s, n) for n in optimizer.get_slot_names() for s in model_vars if optimizer.get_slot(s, n) is not None]
        context_opt_vars = [optimizer.get_slot(s, n) for n in optimizer.get_slot_names() for s in context_agg_vars if optimizer.get_slot(s, n) is not None]

        model_vars += frontend_opt_vars
        context_agg_vars += context_opt_vars

        if FLAGS.load_dir:
            reader = tf.train.NewCheckpointReader(FLAGS.load_dir + ".tf")
            saved_var_map = reader.get_variable_to_shape_map()
            intersect_vars = [k for k in tf.global_variables() if k.name.split(':')[0] in saved_var_map and k.get_shape() == saved_var_map[k.name.split(':')[0]]]
            leftovers = [k for k in tf.global_variables() if k.name.split(':')[0] not in saved_var_map or k.get_shape() != saved_var_map[k.name.split(':')[0]]]
            print("WARNING: Loading pretrained frontend, but not loading: ", map(lambda v: v.name, leftovers))
            frontend_loader = tf.train.Saver(var_list=intersect_vars)

        else:
            frontend_loader = tf.train.Saver(var_list=model_vars)

        frontend_saver = tf.train.Saver(var_list=model_vars)

        if FLAGS.layers2 != '':
            context_saver = tf.train.Saver(context_agg_vars)
            saver = tf.train.Saver(tf.trainable_variables())

        sv = tf.train.Supervisor(logdir=FLAGS.model_dir if FLAGS.model_dir != '' else None,
                                        global_step=global_step,
                                        saver=None,
                                        save_model_secs=0,
                                        save_summaries_secs=0
                                        )

        training_start_time = time.time()
        with sv.managed_session(FLAGS.master, config=tf.ConfigProto(allow_soft_placement=True)) as sess:
            def run_evaluation(eval_batches, use_context_agg, extra_text=""):
                predictions = []
                good_margins = []
                bad_margins = []
                good_maxes = []
                bad_maxes = []
                for b, (eval_label_batch, eval_token_batch, eval_shape_batch, eval_char_batch, eval_seq_len_batch, eval_tok_len_batch, eval_mask_batch) in enumerate(eval_batches):
                    batch_size, batch_seq_len = eval_token_batch.shape

                    # char_lens = np.sum(eval_tok_len_batch, axis=1)
                    # max_char_len = np.max(eval_tok_len_batch)
                    # eval_padded_char_batch = np.zeros((batch_size, max_char_len * batch_seq_len))
                    # for b in range(batch_size):
                    #     char_start = 0
                    #     pad_char_start = 0
                    #     for char_len in eval_tok_len_batch[b]:
                    #         eval_padded_char_batch[b, pad_char_start:pad_char_start + char_len] = eval_char_batch[b][char_start:char_start + char_len]
                    #         pad_char_start += max_char_len
                    #         char_start += char_len
                    char_lens = np.sum(eval_tok_len_batch, axis=1)
                    max_char_len = np.max(eval_tok_len_batch)
                    eval_padded_char_batch = np.zeros((batch_size, max_char_len * batch_seq_len))
                    for b in range(batch_size):
                        char_indices = [item for sublist in [range(i * max_char_len, i * max_char_len + d) for i, d in
                                                             enumerate(eval_tok_len_batch[b])] for item in sublist]
                        eval_padded_char_batch[b, char_indices] = eval_char_batch[b][:char_lens[b]]

                    char_embedding_feeds = {} if FLAGS.char_dim == 0 else {
                        char_embedding_model.input_chars: eval_padded_char_batch,
                        char_embedding_model.batch_size: batch_size,
                        char_embedding_model.max_seq_len: batch_seq_len,
                        char_embedding_model.token_lengths: eval_tok_len_batch,
                        char_embedding_model.max_tok_len: max_char_len
                    }

                    basic_feeds = {
                        model.input_x1: eval_token_batch,
                        model.input_x2: eval_shape_batch,
                        model.input_y: eval_label_batch,
                        model.input_mask: eval_mask_batch,
                        model.max_seq_len: batch_seq_len,
                        model.batch_size: batch_size,
                        model.sequence_lengths: eval_seq_len_batch
                    }

                    context_feeds = {} if not use_context_agg else {
                        context_agg.input_y: eval_label_batch,
                        context_agg.input_mask: eval_mask_batch,
                        context_agg.max_seq_len: batch_seq_len,
                        context_agg.batch_size: batch_size,
                        context_agg.sequence_lengths: eval_seq_len_batch
                    }

                    basic_feeds.update(char_embedding_feeds)
                    total_feeds = basic_feeds.copy()
                    total_feeds.update(context_feeds)

                    if FLAGS.viterbi:
                        if use_context_agg:
                            preds, transition_params = sess.run([context_agg.predictions, context_agg.transition_params], feed_dict=total_feeds)
                        else:
                            preds, transition_params = sess.run([model.predictions, model.transition_params], feed_dict=total_feeds)

                        viterbi_repad = np.empty((batch_size, batch_seq_len))
                        if FLAGS.predict_pad:
                            for batch_idx, (unary_scores, sequence_lens) in enumerate(zip(preds, eval_seq_len_batch)):
                                viterbi_sequence, _ = tf.contrib.crf.viterbi_decode(unary_scores, transition_params)
                                viterbi_repad[batch_idx] = viterbi_sequence
                                # start = 0
                                # for sequence_len in sequence_lens:
                                #     # Remove padding from the scores and tag sequence.
                                #     # unary_scores = unary_scores[pad_width:sequence_len+pad_width]
                                #     unary_scores = unary_scores[start:start + sequence_len + (2 if FLAGS.start_end else 1) * pad_width]
                                #
                                #     # Compute the highest scoring sequence.
                                #     viterbi_sequence, _ = tf.contrib.crf.viterbi_decode(unary_scores, transition_params)
                                #     viterbi_repad[batch_idx, start:start + sequence_len + (2 if FLAGS.start_end else 1) * pad_width] = viterbi_sequence
                                #     start += sequence_len + (2 if FLAGS.start_end else 1) * pad_width
                        else:
                            for batch_idx, (unary_scores, sequence_lens) in enumerate(zip(preds, eval_seq_len_batch)):
                                start = pad_width
                                for sequence_len in sequence_lens:
                                    # Remove padding from the scores and tag sequence.
                                    # unary_scores = unary_scores[pad_width:sequence_len+pad_width]
                                    unary_scores = unary_scores[start:start+sequence_len]

                                    # Compute the highest scoring sequence.
                                    viterbi_sequence, _ = tf.contrib.crf.viterbi_decode(unary_scores, transition_params)
                                    viterbi_repad[batch_idx, start:start+sequence_len] = viterbi_sequence
                                    start += sequence_len + (2 if FLAGS.start_end else 1)*pad_width
                        predictions.append(viterbi_repad)
                    else:
                        if use_context_agg:
                            # preds1 = sess.run([model.predictions],
                            #                  feed_dict={
                            #                      model.input_x1: eval_token_batch,
                            #                      model.input_x2: eval_shape_batch,
                            #                      model.input_x3: eval_char_batch,
                            #                      model.input_y: eval_label_batch,
                            #                      model.input_mask: eval_mask_batch,
                            #                      model.max_seq_len: batch_seq_len,
                            #                      model.batch_size: batch_size,
                            #                      model.sequence_lengths: eval_seq_len_batch
                            #                  }
                            #                  )
                            #
                            # initial_preds = preds1[0]
                            # masked_preds1 = np.multiply(initial_preds, eval_mask_batch)
                            # masked_gold = np.multiply(eval_label_batch, eval_mask_batch)
                            # incorrect_indices = np.where(masked_preds1 != masked_gold)
                            # eval_mask_batch.fill(0)
                            # eval_mask_batch[incorrect_indices] = 1
                            #
                            # preds = sess.run([context_agg.predictions],
                            #     feed_dict={
                            #         model.input_x1: eval_token_batch,
                            #         model.input_x2: eval_shape_batch,
                            #         model.input_x3: eval_char_batch,
                            #         model.input_y: eval_label_batch,
                            #         model.input_mask: eval_mask_batch,
                            #         model.max_seq_len: batch_seq_len,
                            #         model.batch_size: batch_size,
                            #         model.sequence_lengths: eval_seq_len_batch,
                            #
                            #         context_agg.input_y: eval_label_batch,
                            #         context_agg.input_mask: eval_mask_batch,
                            #         context_agg.max_seq_len: batch_seq_len,
                            #         context_agg.batch_size: batch_size,
                            #         context_agg.sequence_lengths: eval_seq_len_batch
                            #     }
                            # )
                            # second_preds = preds[0]
                            # initial_preds[incorrect_indices] = second_preds[incorrect_indices]
                            # predictions.append(initial_preds)

                            # initial_scores, initial_preds = sess.run([model.unflat_scores, model.predictions], feed_dict=basic_feeds)
                            # # initial_preds = preds1[0]
                            # max2 = np.sort(initial_scores, axis=-1)[:,:,-2:]
                            #
                            # # don't bother with entries where we passed margin
                            # max2_margins = max2[:, :, 1] - max2[:, :, 0]
                            # max2_indices = np.where(max2_margins > FLAGS.margin)
                            # incorrect_indices = np.where(max2_margins <= FLAGS.margin)
                            #
                            # max_indices = np.argmax(initial_scores, axis=-1)
                            #
                            # label_indices = batch_seq_len * labels_size * np.arange(batch_size)[:, None] + labels_size * np.arange(batch_seq_len) + eval_label_batch
                            # label_scores_flat = np.reshape(initial_scores, (-1))[label_indices]
                            # label_scores = np.reshape(label_scores_flat, (batch_size, batch_seq_len))
                            # # max2_margins_masked = np.multiply(max2_margins, eval_mask_batch)
                            # correct_margins = max2_margins[np.where((max_indices == eval_label_batch) & (eval_mask_batch == 1.0))]
                            # incorrect_margins = max2_margins[np.where((max_indices != eval_label_batch) & (eval_mask_batch == 1.0))]
                            # correct_maxes = max2[:,:,1][np.where((max_indices == eval_label_batch) & (eval_mask_batch == 1.0))]
                            # incorrect_maxes = max2[:,:,1][np.where((max_indices != eval_label_batch) & (eval_mask_batch == 1.0))]
                            # good_margins.append(correct_margins)
                            # bad_margins.append(incorrect_margins)
                            # good_maxes.append(correct_maxes)
                            # bad_maxes.append(incorrect_maxes)
                            # # print("Avg correct margin: %g" % (np.mean(correct_margins)))
                            # # print("Avg incorrect margin: %g" % (np.mean(incorrect_margins)))
                            #
                            # eval_mask_batch2 = np.copy(eval_mask_batch)
                            # eval_mask_batch2[max2_indices[0], max2_indices[1]] = 0
                            #
                            # this_context_feeds = {
                            #     context_agg.input_y: eval_label_batch,
                            #     context_agg.input_mask: eval_mask_batch2,
                            #     context_agg.max_seq_len: batch_seq_len,
                            #     context_agg.batch_size: batch_size,
                            #     context_agg.sequence_lengths: eval_seq_len_batch
                            # }
                            # this_context_feeds.update(basic_feeds)
                            #
                            # second_preds, second_scores = sess.run([context_agg.predictions, context_agg.unflat_scores], feed_dict=this_context_feeds)
                            #
                            # max2_second = np.sort(second_scores, axis=-1)[:, :, -2:]
                            # max2_margins_second = max2_second[:, :, 1] - max2_second[:, :, 0]
                            # improved_indices = np.where(max2_margins_second > max2_margins)
                            #
                            # # take all the new things that were incorrect before
                            # initial_preds[incorrect_indices] = second_preds[incorrect_indices]
                            #
                            # # take all the new things that increased margin
                            # # initial_preds[improved_indices] = second_preds[improved_indices]
                            #
                            # predictions.append(initial_preds)

                            # basic thing
                            second_preds, second_scores = sess.run([context_agg.predictions, context_agg.unflat_scores], feed_dict=total_feeds)
                            predictions.append(second_preds)

                        else:
                            if FLAGS.model == "cnn-multi":
                                eval_type_batch = np.vectorize(type_int_int_map.__getitem__)(eval_label_batch)
                                eval_bio_batch = np.vectorize(bilou_int_int_map.__getitem__)(eval_label_batch)
                                multi_feeds = {
                                    model.input_x1: eval_token_batch,
                                    model.input_x2: eval_shape_batch,
                                    model.input_y: eval_label_batch,
                                    model.input_y_bio: eval_bio_batch,
                                    model.input_y_type: eval_type_batch,
                                    model.input_mask: eval_mask_batch,
                                    model.max_seq_len: batch_seq_len,
                                    model.batch_size: batch_size,
                                    model.sequence_lengths: eval_seq_len_batch
                                }
                                multi_feeds.update(char_embedding_feeds)
                                preds = sess.run([model.predictions, model.unflat_scores], feed_dict=multi_feeds)

                                # print("pred", [bilou_int_str_map[pred_bio] + "-" + type_int_str_map[pred_label] if type_int_str_map[pred_label] != "O" and bilou_int_str_map[pred_bio] != "O" else "O" for pred_label, pred_bio in zip(preds_label.flatten(), preds_bio.flatten())])
                                # print("gold", map(labels_id_str_map.__getitem__, eval_label_batch.flatten()))
                                #
                                # preds = np.array([labels_str_id_map[bilou_int_str_map[pred_bio] + "-" + type_int_str_map[pred_label] if type_int_str_map[pred_label] != "O" and bilou_int_str_map[pred_bio] != "O" else "O"] for pred_label, pred_bio in zip(preds_label.flatten(), preds_bio.flatten())]).reshape((batch_size, batch_seq_len))
                                # predictions.append(preds)
                            else:
                                preds, scores = sess.run([model.predictions, model.unflat_scores], feed_dict=total_feeds)
                                # max2 = np.sort(scores, axis=-1)[:, :, -2:]
                                #
                                # # don't bother with entries where we passed margin
                                # max2_margins = max2[:, :, 1] - max2[:, :, 0]
                                # max_indices = np.argmax(scores, axis=-1)
                                # correct_margins = max2_margins[np.where((max_indices == eval_label_batch) & (eval_mask_batch == 1.0))]
                                # incorrect_margins = max2_margins[np.where((max_indices != eval_label_batch) & (eval_mask_batch == 1.0))]
                                # correct_maxes = max2[:, :, 1][np.where((max_indices == eval_label_batch) & (eval_mask_batch == 1.0))]
                                # incorrect_maxes = max2[:, :, 1][np.where((max_indices != eval_label_batch) & (eval_mask_batch == 1.0))]
                                # good_margins.append(correct_margins)
                                # bad_margins.append(incorrect_margins)
                                # good_maxes.append(correct_maxes)
                                # bad_maxes.append(incorrect_maxes)
                            predictions.append(preds)

                # merge good/bad margins
                # good_margins = np.concatenate(good_margins)
                # bad_margins = np.concatenate(bad_margins)
                #
                # print
                if good_margins and bad_margins:
                    print("good margins mean: %f" % (np.mean(np.concatenate(good_margins))))
                    print("bad margins mean: %f" % (np.mean(np.concatenate(bad_margins))))
                    print("good maxes mean: %f" % (np.mean(np.concatenate(good_maxes))))
                    print("bad maxes mean: %f" % (np.mean(np.concatenate(bad_maxes))))
                    good_fname = "good_margins.txt"
                    bad_fname = "bad_margins.txt"
                    with open(good_fname, 'w') as f:
                        for v in good_margins:
                            for x in v:
                                print(x, file=f)
                        f.close()
                    with open(bad_fname, 'w') as f:
                        for v in bad_margins:
                            for x in v:
                                print(x, file=f)
                        f.close()

                if FLAGS.print_preds != '':
                    evaluation.print_conlleval_format(FLAGS.print_preds, eval_batches, predictions, labels_id_str_map, vocab_id_str_map, pad_width)

                # print evaluation
                f1_micro, precision = evaluation.segment_eval(eval_batches, predictions, type_set, type_int_int_map,
                                                   labels_id_str_map, vocab_id_str_map,
                                                   outside_idx=map(lambda t: type_set[t] if t in type_set else type_set["O"], outside_set),
                                                   pad_width=pad_width, start_end=FLAGS.start_end,
                                                   extra_text="Segment evaluation %s:" % extra_text)
                # evaluation.token_eval(dev_batches, predictions, type_set, type_int_int_map, outside_idx=type_set["O"],
                #                       pad_width=pad_width, extra_text="Token evaluation %s:" % extra_text)
                # evaluation.boundary_eval(eval_batches, predictions, bilou_set, bilou_int_int_map,
                #                          outside_idx=bilou_set["O"], pad_width=pad_width,
                #                          extra_text="Boundary evaluation %s: " % extra_text)

                return f1_micro, precision

            threads = tf.train.start_queue_runners(sess=sess)
            log_every = int(max(100, num_train_examples / 5))

            if FLAGS.load_dir != '':
                print("Deserializing model: " + FLAGS.load_dir + ".tf")
                frontend_loader.restore(sess, FLAGS.load_dir + ".tf")
            if FLAGS.context_load_dir != '':
                print("Deserializing model: " + FLAGS.context_load_dir + ".tf")
                context_saver.restore(sess, FLAGS.context_load_dir + ".tf")
            if FLAGS.all_load_dir != '':
                print("Deserializing model: " + FLAGS.all_load_dir + ".tf")
                saver.restore(sess, FLAGS.all_load_dir + ".tf")

            def get_dev_batches(seq_batcher):
                batches = []
                # load all the dev batches into memory
                done = False
                while not done:
                    try:
                        dev_batch = sess.run(seq_batcher.next_batch_op)
                        dev_label_batch, dev_token_batch, dev_shape_batch, dev_char_batch, dev_seq_len_batch, dev_tok_len_batch = dev_batch
                        mask_batch = np.zeros(dev_token_batch.shape)
                        if FLAGS.predict_pad:
                            # np.add(np.sum(seq_len_batch, axis=1), (2 if FLAGS.start_end else 1) * pad_width * (num_sentences_batch + (0 if FLAGS.start_end else 1)))
                            actual_seq_lens = np.add(np.sum(dev_seq_len_batch, axis=1),
                                                     (2 if FLAGS.start_end else 1) * pad_width * (
                                                     (dev_seq_len_batch != 0).sum(axis=1) + (
                                                     0 if FLAGS.start_end else 1)))
                            for i, seq_len in enumerate(actual_seq_lens):
                                mask_batch[i, :seq_len] = 1
                        else:
                            for i, seq_lens in enumerate(dev_seq_len_batch):
                                start = pad_width
                                for seq_len in seq_lens:
                                    mask_batch[i, start:start + seq_len] = 1
                                    start += seq_len + (2 if FLAGS.start_end else 1) * pad_width
                        batches.append((dev_label_batch, dev_token_batch, dev_shape_batch, dev_char_batch,
                                            dev_seq_len_batch, dev_tok_len_batch, mask_batch))
                    except:
                        done = True
                return batches
            dev_batches = get_dev_batches(dev_batcher)
            if FLAGS.ontonotes:
                domain_batches = {domain: get_dev_batches(domain_batcher)
                                  for domain, domain_batcher in domain_dev_batchers.iteritems()}

            train_batches = []
            if FLAGS.train_eval:
                # load all the train batches into memory
                done = False
                while not done:
                    try:
                        train_batch = sess.run(train_eval_batcher.next_batch_op)
                        train_label_batch, train_token_batch, train_shape_batch, train_char_batch, train_seq_len_batch, train_tok_len_batch = train_batch
                        mask_batch = np.zeros(train_token_batch.shape)
                        if FLAGS.predict_pad:
                            # actual_seq_lens = np.add(np.sum(train_seq_len_batch, axis=1), 2 * pad_width * (train_seq_len_batch != 0).sum(axis=1))
                            actual_seq_lens = np.add(np.sum(train_seq_len_batch, axis=1), (2 if FLAGS.start_end else 1) * pad_width * ((train_seq_len_batch != 0).sum(axis=1) + (0 if FLAGS.start_end else 1)))
                            for i, seq_len in enumerate(actual_seq_lens):
                                mask_batch[i, :seq_len] = 1
                        else:
                            for i, seq_lens in enumerate(train_seq_len_batch):
                                start = pad_width
                                for seq_len in seq_lens:
                                    mask_batch[i, start:start + seq_len] = 1
                                    start += seq_len + (2 if FLAGS.start_end else 1) * pad_width
                        train_batches.append((train_label_batch, train_token_batch, train_shape_batch, train_char_batch, train_seq_len_batch, train_tok_len_batch, mask_batch))
                    except Exception as e:
                        done = True
            if FLAGS.memmap_train:
                train_batcher.load_and_bucket_data(sess)

            def train(max_epochs, best_score, best_precision, model_hidden_drop, model_input_drop, until_convergence, update_frontend, update_context, max_lower=6, min_iters=20):
                print("Training on %d sentences (%d examples); front end: %r; context: %r" % (num_train_examples, num_train_examples, update_frontend, update_context))
                start_time = time.time()
                train_batcher._step = 1.0
                converged = False
                examples = 0
                log_every_running = log_every
                epoch_loss = 0.0
                num_lower = 0
                training_iteration = 0
                speed_num = 0.0
                speed_denom = 0.0
                while not sv.should_stop() and training_iteration < max_epochs and not (until_convergence and converged):
                    # evaluate
                    if examples >= num_train_examples:
                        training_iteration += 1

                        if FLAGS.train_eval:
                            run_evaluation(train_batches, update_context, "TRAIN (iteration %d)" % training_iteration)
                        print()
                        f1_micro, precision = run_evaluation(dev_batches, update_context, "TEST (iteration %d)" % training_iteration)
                        print("Avg training speed: %f examples/second" % (speed_num/speed_denom))

                        # keep track of running best / convergence heuristic
                        if f1_micro > best_score:
                            best_score = f1_micro
                            num_lower = 0
                            if FLAGS.model_dir != '' and best_score > FLAGS.save_min:
                                if update_frontend and not update_context:
                                    save_path = frontend_saver.save(sess, FLAGS.model_dir + "-frontend.tf")
                                    print("Serialized model: %s" % save_path)
                                elif update_context and not update_frontend:
                                    save_path = context_saver.save(sess, FLAGS.model_dir + "-context.tf")
                                    print("Serialized model: %s" % save_path)
                                else:
                                    save_path = saver.save(sess, FLAGS.model_dir + ".tf")
                                    print("Serialized model: %s" % save_path)
                        else:
                            num_lower += 1
                        if num_lower > max_lower and training_iteration > min_iters:
                            converged = True

                        # if precision > best_precision:
                        #     best_precision = precision
                        #     if FLAGS.model_dir != '':
                        #         if update_frontend and not update_context:
                        #             save_path = frontend_saver.save(sess, FLAGS.model_dir + "-frontend-prec.tf")
                        #             print("Serialized model: %s" % save_path)
                        #         elif update_context and not update_frontend:
                        #             save_path = context_saver.save(sess, FLAGS.model_dir + "-context-prec.tf")
                        #             print("Serialized model: %s" % save_path)
                        #         else:
                        #             save_path = saver.save(sess, FLAGS.model_dir + "-prec.tf")
                        #             print("Serialized model: %s" % save_path)

                        # update per-epoch variables
                        log_every_running = log_every
                        examples = 0
                        epoch_loss = 0.0
                        start_time = time.time()

                    if examples > log_every_running:
                        speed_denom += time.time()-start_time
                        speed_num += examples
                        evaluation.print_training_error(examples, start_time, [epoch_loss], train_batcher._step)
                        log_every_running += log_every

                    # Training iteration

                    label_batch, token_batch, shape_batch, char_batch, seq_len_batch, tok_lengths_batch = \
                        train_batcher.next_batch() if FLAGS.memmap_train else sess.run(train_batcher.next_batch_op)

                    # make mask out of seq lens
                    # print("batch shape: ", token_batch.shape)
                    batch_size, batch_seq_len = token_batch.shape

                    # print(batch_seq_len)
                    # print("label",label_batch)
                    # print("token", token_batch)
                    # print("char", char_batch)
                    # print(tok_lengths_batch)

                    char_lens = np.sum(tok_lengths_batch, axis=1)
                    max_char_len = np.max(tok_lengths_batch)
                    padded_char_batch = np.zeros((batch_size, max_char_len * batch_seq_len))
                    for b in range(batch_size):
                        char_indices = [item for sublist in [range(i * max_char_len, i * max_char_len + d) for i, d in
                                                             enumerate(tok_lengths_batch[b])] for item in sublist]
                        padded_char_batch[b, char_indices] = char_batch[b][:char_lens[b]]

                    # try:
                    max_sentences = max(map(len, seq_len_batch))
                    new_seq_len_batch = np.zeros((batch_size, max_sentences))
                    for i, seq_len_list in enumerate(seq_len_batch):
                        new_seq_len_batch[i,:len(seq_len_list)] = seq_len_list
                    seq_len_batch = new_seq_len_batch
                    num_sentences_batch = np.sum(seq_len_batch != 0, axis=1)

                    # except Exception as e:
                    #     print(batch_seq_len)
                    #     print("label",label_batch)
                    #     print("token", token_batch)
                    #     print("char", char_batch)
                    #     print(tok_lengths_batch)
                    #     print(seq_len_batch)

                    # print(seq_len_batch)
                    # print(num_sentences_batch)

                    mask_batch = np.zeros((batch_size, batch_seq_len)).astype("int")
                    if FLAGS.predict_pad:
                        actual_seq_lens = np.add(np.sum(seq_len_batch, axis=1), (2 if FLAGS.start_end else 1) * pad_width * (num_sentences_batch + (0 if FLAGS.start_end else 1)))
                        for i, seq_len in enumerate(actual_seq_lens):
                            mask_batch[i, :seq_len] = 1
                    else:
                        for i, seq_lens in enumerate(seq_len_batch):
                            start = pad_width
                            for seq_len in seq_lens:
                                mask_batch[i, start:start+seq_len] = 1
                                start += seq_len + (2 if FLAGS.start_end else 1)*pad_width
                    examples += batch_size

                    # print(batch_seq_len)
                    # print(tok_lengths_batch.shape)
                    # print(tok_lengths_batch)
                    # print(np.reshape(tok_lengths_batch, (batch_size*batch_seq_len)))
                    # print(padded_char_batch.shape)
                    # print(np.reshape(padded_char_batch, (batch_size*batch_seq_len, max_char_len)))

                    # print("input_x1", token_batch)
                    # print("input_x1_sample_pad", input_x1_sample_pad)

                    # apply word dropout
                    # create word dropout mask
                    word_probs = np.random.random(token_batch.shape)
                    drop_indices = np.where((word_probs > FLAGS.word_dropout) & (token_batch != vocab_str_id_map["<PAD>"]))
                    token_batch[drop_indices[0], drop_indices[1]] = vocab_str_id_map["<OOV>"]

                    # apply sentence dropout
                    if FLAGS.sentence_dropout < 1.0 and FLAGS.documents:
                        # if FLAGS.regularize_pad_penalty != 0.0:
                        #     print("Sentence dropout breaks pad penalty; Exiting")
                        #     sys.exit(1)

                        max_sampled_seq_len = batch_seq_len
                        input_x1_sample_pad = np.copy(token_batch)
                        input_x2_sample_pad = np.copy(shape_batch)
                        # input_x3_sample_pad = np.copy(char_batch)
                        input_mask_sample_pad = np.copy(mask_batch)

                        sent_probs = np.random.random((batch_size, np.max(num_sentences_batch)))
                        # sent_mask = np.zeros((batch_size, np.max(num_sentences_batch)))
                        # for i, num_sents in enumerate(num_sentences_batch):
                        #     sent_mask[i, :num_sents] = 1
                        # sent_probs = np.multiply(sent_probs, sent_mask)
                        sent_drop_mask = sent_probs > FLAGS.sentence_dropout
                        # print(sent_drop_mask)
                        for i, seq_lens in enumerate(seq_len_batch):
                            if not np.all(sent_drop_mask[i]):
                                start = pad_width
                                # print("before", token_batch[i])
                                # print("before", mask_batch[i])
                                for j, seq_len in enumerate(seq_lens):
                                    if sent_drop_mask[i, j]:
                                        input_x1_sample_pad[i, start:start+seq_len] = vocab_str_id_map["<PAD>"]
                                        input_x2_sample_pad[i, start:start+seq_len] = shape_str_id_map["<PAD>"]
                                        input_mask_sample_pad[i, start:start+seq_len] = 0
                                    start += seq_len + (2 if FLAGS.start_end else 1) * pad_width
                                # print("after", token_batch[i])
                                # print("after", mask_batch[i])
                        mask_batch = input_mask_sample_pad

                    else:
                        # sample padding
                        # sample an amount of padding to add for each sentence

                        max_sampled_seq_len = batch_seq_len + (np.max(num_sentences_batch) + 1) * FLAGS.max_additional_pad
                        input_x1_sample_pad = np.empty((batch_size, max_sampled_seq_len))
                        input_x2_sample_pad = np.empty((batch_size, max_sampled_seq_len))
                        # input_x3_sample_pad = np.empty((batch_size, max_sampled_seq_len))
                        input_mask_sample_pad = np.zeros((batch_size, max_sampled_seq_len))
                        if FLAGS.regularize_pad_penalty != 0.0:
                            input_x1_sample_pad.fill(vocab_str_id_map["<PAD>"])
                            input_x2_sample_pad.fill(shape_str_id_map["<PAD>"])
                            # input_x3_sample_pad.fill(char_str_id_map["<PAD>"])

                            for i, seq_lens in enumerate(seq_len_batch):
                                pad_start = sample_pad_size()
                                actual_start = pad_width
                                for seq_len in seq_lens:
                                    input_x1_sample_pad[i, pad_start:pad_start+seq_len] = token_batch[i, actual_start:actual_start+seq_len]
                                    input_x2_sample_pad[i, pad_start:pad_start+seq_len] = shape_batch[i, actual_start:actual_start+seq_len]
                                    # input_x3_sample_pad[i, pad_start:pad_start+seq_len] = char_batch[i, actual_start:actual_start+seq_len]
                                    input_mask_sample_pad[i, pad_start:pad_start + seq_len] = 1
                                    sampled_pad_size = sample_pad_size()
                                    pad_start += seq_len + sampled_pad_size
                                    actual_start += seq_len + (2 if FLAGS.start_end else 1)*pad_width

                    char_embedding_feeds = {} if FLAGS.char_dim == 0 else {
                        char_embedding_model.input_chars: padded_char_batch,
                        char_embedding_model.batch_size: batch_size,
                        char_embedding_model.max_seq_len: batch_seq_len,
                        char_embedding_model.token_lengths: tok_lengths_batch,
                        char_embedding_model.max_tok_len: max_char_len,
                        char_embedding_model.input_dropout_keep_prob: FLAGS.char_input_dropout
                    }

                    if FLAGS.model == "cnn" or FLAGS.model == "cnn-fast":
                        cnn_feeds = {
                                model.input_x1: token_batch,
                                model.input_x2: shape_batch,
                                model.input_y: label_batch,
                                model.input_mask: mask_batch,
                                model.max_seq_len: batch_seq_len,
                                model.sequence_lengths: seq_len_batch,
                                model.batch_size: batch_size,
                                model.hidden_dropout_keep_prob: model_hidden_drop,
                                model.input_dropout_keep_prob: model_input_drop,
                                model.middle_dropout_keep_prob: FLAGS.middle_dropout,
                                model.l2_penalty: FLAGS.l2,
                                model.pad_penalty: FLAGS.regularize_pad_penalty,
                                model.drop_penalty: FLAGS.regularize_drop_penalty,
                                model.training: True,
                                model.sample_pad_max_seq_len: max_sampled_seq_len,
                                model.input_x1_sample_pad: input_x1_sample_pad,
                                model.input_x2_sample_pad: input_x2_sample_pad,
                                # model.input_x3_sample_pad: input_x3_sample_pad,
                                model.input_mask_sample_pad: input_mask_sample_pad
                            }
                        cnn_feeds.update(char_embedding_feeds)
                        if update_frontend and not update_context:
                            _,  loss = sess.run([train_op, model.loss], feed_dict=cnn_feeds)

                        elif update_context and not update_frontend:

                                # # train with gold
                                # # preds1 = sess.run([model.predictions],
                                # #                   feed_dict={
                                # #                       model.input_x1: token_batch,
                                # #                       model.input_x2: shape_batch,
                                # #                       model.input_x3: char_batch,
                                # #                       model.input_y: label_batch,
                                # #                       model.input_mask: mask_batch,
                                # #                       model.max_seq_len: batch_seq_len,
                                # #                       model.batch_size: batch_size,
                                # #                       model.sequence_lengths: seq_len_batch
                                # #                   }
                                # #                   )
                                # #
                                # # initial_preds = preds1[0]
                                # # masked_preds1 = np.multiply(initial_preds, mask_batch)
                                # # masked_gold = np.multiply(label_batch, mask_batch)
                                # # incorrect_indices = np.where(masked_preds1 != masked_gold)
                                # # mask_batch.fill(0)
                                # # mask_batch[incorrect_indices] = 1
                                # #
                                # # # print("wrong1: ", np.sum(mask_batch))
                                # #
                                # # _, loss = sess.run([context_agg_train_op, context_agg.loss],
                                # #     feed_dict={
                                # #         model.input_x1: token_batch,
                                # #         model.input_x2: shape_batch,
                                # #         model.input_x3: char_batch,
                                # #         model.input_y: label_batch,
                                # #         model.input_mask: mask_batch,
                                # #         model.max_seq_len: batch_seq_len,
                                # #         model.sequence_lengths: seq_len_batch,
                                # #         model.batch_size: batch_size,
                                # #         # model.hidden_dropout_keep_prob: model_hidden_drop,
                                # #         # model.input_dropout_keep_prob: model_input_drop,
                                # #         # model.middle_dropout_keep_prob: FLAGS.middle_dropout,
                                # #         # model.l2_penalty: FLAGS.l2,
                                # #         # model.pad_penalty: FLAGS.regularize_pad_penalty,
                                # #         # model.drop_penalty: FLAGS.regularize_drop_penalty,
                                # #         # model.training: True,
                                # #
                                # #         model.sample_pad_max_seq_len: max_sampled_seq_len,
                                # #         model.input_x1_sample_pad: input_x1_sample_pad,
                                # #         model.input_x2_sample_pad: input_x2_sample_pad,
                                # #         model.input_x3_sample_pad: input_x3_sample_pad,
                                # #         model.input_mask_sample_pad: input_mask_sample_pad,
                                # #
                                # #         context_agg.input_y: label_batch,
                                # #         context_agg.input_mask: mask_batch,
                                # #         context_agg.max_seq_len: batch_seq_len,
                                # #         context_agg.batch_size: batch_size,
                                # #         context_agg.hidden_dropout_keep_prob: FLAGS.hidden2_dropout,
                                # #         context_agg.input_dropout_keep_prob: FLAGS.input2_dropout,
                                # #         # context_agg.middle_dropout_keep_prob: FLAGS.middle_dropout,
                                # #         context_agg.sequence_lengths: seq_len_batch,
                                # #         context_agg.input_mask_sample_pad: input_mask_sample_pad,
                                # #         context_agg.sample_pad_max_seq_len: max_sampled_seq_len,
                                # #         context_agg.drop_penalty: FLAGS.context_regularize_drop_penalty
                                # #     }
                                # #     )
                                #
                                # # train with max2
                                # initial_scores, initial_preds = sess.run([model.unflat_scores, model.predictions],
                                #                   feed_dict={
                                #                       model.input_x1: token_batch,
                                #                       model.input_x2: shape_batch,
                                #                       model.input_x3: char_batch,
                                #                       model.input_y: label_batch,
                                #                       model.input_mask: mask_batch,
                                #                       model.max_seq_len: batch_seq_len,
                                #                       model.batch_size: batch_size,
                                #                       model.sequence_lengths: seq_len_batch
                                #                   }
                                #                   )
                                #
                                # # max2
                                # max2 = np.sort(initial_scores, axis=-1)[:, :, -2:]
                                #
                                # # don't bother with entries where we passed margin
                                # max2_margins = max2[:, :, 1] - max2[:, :, 0]
                                # max2_indices = np.where(max2_margins > FLAGS.margin)
                                #
                                # # print("Avg correct margin: %g" % (np.mean(correct_margins)))
                                # # print("Avg incorrect margin: %g" % (np.mean(incorrect_margins)))
                                #
                                # mask_batch2 = np.copy(mask_batch)
                                # mask_batch2[max2_indices[0], max2_indices[1]] = 0
                                #
                                # # use labels
                                # # initial_scores = preds1[0]
                                # #
                                # # label_indices = batch_seq_len * labels_size * np.arange(batch_size)[:, None] + labels_size * np.arange(batch_seq_len) + label_batch
                                # #
                                # # # adjust for margin
                                # # np.reshape(initial_scores, (-1))[label_indices] -= FLAGS.margin
                                # # # print(np.mean(initial_preds))
                                # # # label_preds = np.reshape(initial_preds, (-1))[label_indices]
                                # # max_preds = np.max(initial_scores, axis=-1)
                                # # label_scores = np.reshape(initial_scores, (-1))[label_indices]
                                # # non_violators = np.where(label_scores > max_preds)
                                # # mask_batch2 = np.copy(mask_batch)
                                # # mask_batch2[non_violators] = 0
                                context_feeds = {
                                    context_agg.input_y: label_batch,
                                    context_agg.input_mask: mask_batch,
                                    context_agg.max_seq_len: batch_seq_len,
                                    context_agg.batch_size: batch_size,
                                    context_agg.hidden_dropout_keep_prob: FLAGS.hidden2_dropout,
                                    context_agg.input_dropout_keep_prob: FLAGS.input2_dropout,
                                    # context_agg.middle_dropout_keep_prob: FLAGS.middle_dropout,
                                    context_agg.sequence_lengths: seq_len_batch,
                                    context_agg.input_mask_sample_pad: input_mask_sample_pad,
                                    context_agg.sample_pad_max_seq_len: max_sampled_seq_len,
                                    context_agg.drop_penalty: FLAGS.context_regularize_drop_penalty
                                }
                                context_feeds.update(cnn_feeds)
                                _, loss = sess.run([context_agg_train_op, context_agg.loss], feed_dict=context_feeds)
                        else:
                            # _, _, loss1, loss2 = sess.run(
                            #     [train_op, all_train_op, context_agg.loss, model.loss],
                            #     feed_dict={
                            #         model.input_x1: token_batch,
                            #         model.input_x2: shape_batch,
                            #         model.input_x3: char_batch,
                            #         model.input_y: label_batch,
                            #         model.input_mask: mask_batch,
                            #         model.max_seq_len: batch_seq_len,
                            #         model.sequence_lengths: seq_len_batch,
                            #         model.batch_size: batch_size,
                            #         model.training: True,
                            #         model.l2_penalty: FLAGS.l2,
                            #         model.pad_penalty: FLAGS.regularize_pad_penalty,
                            #         model.drop_penalty: FLAGS.regularize_drop_penalty,
                            #
                            #         model.hidden_dropout_keep_prob: model_hidden_drop,
                            #         model.input_dropout_keep_prob: model_input_drop,
                            #         model.middle_dropout_keep_prob: FLAGS.middle_dropout,
                            #
                            #         model.sample_pad_max_seq_len: max_sampled_seq_len,
                            #         model.input_x1_sample_pad: input_x1_sample_pad,
                            #         model.input_x2_sample_pad: input_x2_sample_pad,
                            #         model.input_x3_sample_pad: input_x3_sample_pad,
                            #         model.input_mask_sample_pad: input_mask_sample_pad,
                            #
                            #         context_agg.input_y: label_batch,
                            #         context_agg.input_mask: mask_batch,
                            #         context_agg.max_seq_len: batch_seq_len,
                            #         context_agg.batch_size: batch_size,
                            #         context_agg.hidden_dropout_keep_prob: FLAGS.hidden2_dropout,
                            #         context_agg.input_dropout_keep_prob: FLAGS.input2_dropout,
                            #         # context_agg.middle_dropout_keep_prob: FLAGS.middle_dropout,
                            #         context_agg.sequence_lengths: seq_len_batch,
                            #         context_agg.input_mask_sample_pad: input_mask_sample_pad,
                            #         context_agg.sample_pad_max_seq_len: max_sampled_seq_len,
                            #         context_agg.drop_penalty: FLAGS.context_regularize_drop_penalty
                            #
                            #     }
                            # )
                            # loss = loss1 + loss2
                                # [train_op, context_agg_train_op, context_agg.loss, model.loss],
                                # feed_dict={
                                #     model.input_x1: token_batch,
                                #     model.input_x2: shape_batch,
                                #     model.input_x3: char_batch,
                                #     model.input_y: label_batch,
                                #     model.input_mask: mask_batch,
                                #     model.max_seq_len: batch_seq_len,
                                #     model.sequence_lengths: seq_len_batch,
                                #     model.batch_size: batch_size,
                                #     model.training: True,
                                #     model.l2_penalty: FLAGS.l2,
                                #     model.pad_penalty: FLAGS.regularize_pad_penalty,
                                #     model.drop_penalty: FLAGS.regularize_drop_penalty,
                                #
                                #     model.hidden_dropout_keep_prob: model_hidden_drop,
                                #     model.input_dropout_keep_prob: model_input_drop,
                                #     model.middle_dropout_keep_prob: FLAGS.middle_dropout,
                                #
                                #     model.sample_pad_max_seq_len: max_sampled_seq_len,
                                #     model.input_x1_sample_pad: input_x1_sample_pad,
                                #     model.input_x2_sample_pad: input_x2_sample_pad,
                                #     model.input_x3_sample_pad: input_x3_sample_pad,
                                #     model.input_mask_sample_pad: input_mask_sample_pad,
                                #
                                #     context_agg.input_y: label_batch,
                                #     context_agg.input_mask: mask_batch,
                                #     context_agg.max_seq_len: batch_seq_len,
                                #     context_agg.batch_size: batch_size,
                                #     context_agg.hidden_dropout_keep_prob: FLAGS.hidden2_dropout,
                                #     context_agg.input_dropout_keep_prob: FLAGS.input2_dropout,
                                #     # context_agg.middle_dropout_keep_prob: FLAGS.middle_dropout,
                                #     context_agg.sequence_lengths: seq_len_batch,
                                #     context_agg.input_mask_sample_pad: input_mask_sample_pad,
                                #     context_agg.sample_pad_max_seq_len: max_sampled_seq_len,
                                #     context_agg.drop_penalty: FLAGS.context_regularize_drop_penalty
                                #
                                # }
                                # )

                            _, loss1, scores = sess.run([train_op, model.loss, model.unflat_scores], feed_dict=cnn_feeds)

                            # max2
                            # max2 = np.sort(scores, axis=-1)[:, :, -2:]
                            #
                            # # don't bother with entries where we passed margin
                            # max2_margins = max2[:, :, 1] - max2[:, :, 0]
                            # max2_indices = np.where(max2_margins > FLAGS.margin)
                            #
                            # # print("Avg correct margin: %g" % (np.mean(correct_margins)))
                            # # print("Avg incorrect margin: %g" % (np.mean(incorrect_margins)))
                            #
                            # mask_batch2 = np.copy(mask_batch)
                            # mask_batch2[max2_indices[0], max2_indices[1]] = 0

                            # # use labels

                            label_indices = batch_seq_len * labels_size * np.arange(batch_size)[:, None] + labels_size * np.arange(batch_seq_len) + label_batch

                            # adjust for margin
                            np.reshape(scores, (-1))[label_indices] -= FLAGS.margin
                            # print(np.mean(initial_preds))
                            # label_preds = np.reshape(initial_preds, (-1))[label_indices]
                            scores_augmented = np.copy(scores)
                            np.reshape(scores_augmented, -1)[label_indices] = np.NINF
                            max_preds = np.max(scores_augmented, axis=-1)
                            label_scores = np.reshape(scores, (-1))[label_indices]
                            non_violators = np.where(label_scores > max_preds)
                            mask_batch2 = np.copy(mask_batch)
                            mask_batch2[non_violators] = 0

                            context_feeds = {
                                context_agg.model_scores: scores,
                                context_agg.reuse_scores: True,

                                context_agg.input_y: label_batch,
                                context_agg.input_mask: mask_batch2,
                                context_agg.max_seq_len: batch_seq_len,
                                context_agg.batch_size: batch_size,
                                context_agg.hidden_dropout_keep_prob: FLAGS.hidden2_dropout,
                                context_agg.input_dropout_keep_prob: FLAGS.input2_dropout,
                                # context_agg.middle_dropout_keep_prob: FLAGS.middle_dropout,
                                context_agg.sequence_lengths: seq_len_batch,
                                context_agg.input_mask_sample_pad: input_mask_sample_pad,
                                context_agg.sample_pad_max_seq_len: max_sampled_seq_len,
                                context_agg.drop_penalty: FLAGS.context_regularize_drop_penalty
                            }
                            context_feeds.update(cnn_feeds)

                            _, loss2 = sess.run([all_train_op, context_agg.loss], feed_dict=context_feeds)
                            loss = loss1 + loss2

                    elif FLAGS.model == "cnn-multi":
                        type_batch = np.vectorize(type_int_int_map.__getitem__)(label_batch)
                        bio_batch = np.vectorize(bilou_int_int_map.__getitem__)(label_batch)
                        feed_multi = {
                            model.input_x1: token_batch,
                            model.input_x2: shape_batch,
                            model.input_y: label_batch,
                            model.input_y_bio: bio_batch,
                            model.input_y_type: type_batch,
                            model.input_mask: mask_batch,
                            model.max_seq_len: batch_seq_len,
                            model.sequence_lengths: seq_len_batch,
                            model.batch_size: batch_size,
                            model.hidden_dropout_keep_prob: model_hidden_drop,
                            model.input_dropout_keep_prob: model_input_drop,
                            model.middle_dropout_keep_prob: FLAGS.middle_dropout,
                            model.l2_penalty: FLAGS.l2,
                            model.pad_penalty: FLAGS.regularize_pad_penalty,
                            model.drop_penalty: FLAGS.regularize_drop_penalty,
                            model.training: True,

                            model.sample_pad_max_seq_len: max_sampled_seq_len,
                            model.input_x1_sample_pad: input_x1_sample_pad,
                            model.input_x2_sample_pad: input_x2_sample_pad,
                            # model.input_x3_sample_pad: input_x3_sample_pad,
                            model.input_mask_sample_pad: input_mask_sample_pad
                        }
                        _, loss = sess.run([train_op, model.loss], feed_dict=feed_multi)
                    elif FLAGS.model == "bilstm":
                        lstm_feed = {
                            model.input_x1: token_batch,
                            model.input_x2: shape_batch,
                            model.input_y: label_batch,
                            model.input_mask: mask_batch,
                            model.sequence_lengths: seq_len_batch,
                            model.max_seq_len: batch_seq_len,
                            model.batch_size: batch_size,
                            model.hidden_dropout_keep_prob: FLAGS.hidden_dropout,
                            model.middle_dropout_keep_prob: FLAGS.middle_dropout,
                            model.input_dropout_keep_prob: FLAGS.input_dropout,
                            model.l2_penalty: FLAGS.l2,
                            model.drop_penalty: FLAGS.regularize_drop_penalty
                        }
                        lstm_feed.update(char_embedding_feeds)
                        _, loss = sess.run([train_op, model.loss], feed_dict=lstm_feed)
                    epoch_loss += loss
                    train_batcher._step += 1
                return best_score, training_iteration, speed_num/speed_denom

            if FLAGS.evaluate_only:
                if FLAGS.train_eval:
                    run_evaluation(train_batches, FLAGS.layers2 != '', "(train)")
                print()
                run_evaluation(dev_batches, FLAGS.layers2 != '', "(test)")
                if FLAGS.ontonotes:
                    for domain, domain_batches in domain_batches.iteritems():
                        print()
                        run_evaluation(domain_batches, FLAGS.layers2 != '', "(test - domain: %s)" % domain)

            else:
                best_score = 0
                total_iterations = 0

                # always train the front-end unless load dir was passed
                if FLAGS.load_dir == '' or (FLAGS.load_dir != '' and FLAGS.layers2 == ''):
                    best_score, training_iteration, train_speed = train(FLAGS.max_epochs, 0.0, 0.0,
                                                           FLAGS.hidden_dropout, FLAGS.input_dropout,
                                                           until_convergence=FLAGS.until_convergence,
                                                           update_context=False, update_frontend=True)
                    total_iterations += training_iteration
                    if FLAGS.model_dir:
                        print("Deserializing model: " + FLAGS.model_dir + "-frontend.tf")
                        frontend_saver.restore(sess, FLAGS.model_dir + "-frontend.tf")

                # if we passed in context agg info, fix frontend and train that
                if FLAGS.layers2 != '' and FLAGS.context_load_dir == '':
                    best_score, training_iteration, train_speed = train(FLAGS.max_context_epochs, 0.0, 0.0,
                                                           FLAGS.hidden_dropout_context, FLAGS.input_dropout_context,
                                                           until_convergence=FLAGS.until_convergence,
                                                           update_context=True, update_frontend=False)
                    total_iterations += training_iteration
                    print("Deserializing model: " + FLAGS.model_dir + "-context.tf")
                    context_saver.restore(sess, FLAGS.model_dir + "-context.tf")

                    # if we want fine-tuning, do that now
                    if FLAGS.update_frontend:
                        best_score, training_iteration, train_speed = train(FLAGS.max_finetune_epochs, 0.0, 0.0,
                                                               FLAGS.hidden_dropout, FLAGS.input_dropout,
                                                               until_convergence=FLAGS.until_convergence,
                                                               update_context=True, update_frontend=True)
                        total_iterations += training_iteration
                elif FLAGS.layers2 != '':
                    best_score, training_iteration, train_speed = train(FLAGS.max_finetune_epochs, 0.0, 0.0,
                                                           FLAGS.hidden_dropout, FLAGS.input_dropout,
                                                           until_convergence=FLAGS.until_convergence,
                                                           update_context=True, update_frontend=True)
                    total_iterations += training_iteration

            sv.coord.request_stop()
            sv.coord.join(threads)
            sess.close()

            total_time = time.time()-training_start_time
            if FLAGS.evaluate_only:
                print("Testing time: %d seconds" % (total_time))
            else:
                print("Training time: %d minutes, %d iterations (%3.2f minutes/iteration)" % (total_time/60, total_iterations, total_time/(60*total_iterations)))
                print("Avg training speed: %f examples/second" % (train_speed))
                print("Best dev F1: %2.2f" % (best_score*100))

if __name__ == '__main__':
    tf.app.flags.DEFINE_string('train_dir', '', 'directory containing preprocessed training data')
    tf.app.flags.DEFINE_string('dev_dir', '', 'directory containing preprocessed dev data')
    tf.app.flags.DEFINE_string('test_dir', '', 'directory containing preprocessed test data')
    tf.app.flags.DEFINE_string('maps_dir', '', 'directory containing data intmaps')

    tf.app.flags.DEFINE_string('model_dir', '', 'save model to this dir (if empty do not save)')
    tf.app.flags.DEFINE_string('load_dir', '', 'load model from this dir (if empty do not load)')
    tf.app.flags.DEFINE_string('context_load_dir', '', 'load context model from this dir (if empty do not load)')
    tf.app.flags.DEFINE_string('all_load_dir', '', 'load model from this dir (if empty do not load)')


    tf.app.flags.DEFINE_string('optimizer', 'adam', 'optimizer to use')
    tf.app.flags.DEFINE_string('master', '', 'use for Supervisor')
    tf.app.flags.DEFINE_string('model', 'cnn', 'which model to use [cnn, seq2seq, lstm, bilstm]')
    tf.app.flags.DEFINE_integer('filter_size', 3, "filter size")

    tf.app.flags.DEFINE_float('lr', 0.001, 'learning rate')
    tf.app.flags.DEFINE_float('l2', 0.0, 'l2 penalty')
    tf.app.flags.DEFINE_float('beta1', 0.9, 'beta1')
    tf.app.flags.DEFINE_float('beta2', 0.999, 'beta2')
    tf.app.flags.DEFINE_float('epsilon', 1e-8, 'epsilon')

    tf.app.flags.DEFINE_float('hidden_dropout_context', 1.0, 'hidden layer dropout rate when training context')
    tf.app.flags.DEFINE_float('input_dropout_context', 1.0, 'input layer (word embedding) dropout rate when training context')


    tf.app.flags.DEFINE_float('hidden_dropout', .75, 'hidden layer dropout rate')
    tf.app.flags.DEFINE_float('hidden2_dropout', .75, 'hidden layer 2 dropout rate')
    tf.app.flags.DEFINE_float('input2_dropout', .75, 'input layer 2 dropout rate')

    tf.app.flags.DEFINE_float('input_dropout', 1.0, 'input layer (word embedding) dropout rate')
    tf.app.flags.DEFINE_float('middle_dropout', 1.0, 'middle layer dropout rate')
    tf.app.flags.DEFINE_float('word_dropout', 1.0, 'whole-word (-> oov) dropout rate')
    tf.app.flags.DEFINE_float('sentence_dropout', 1.0, 'whole-sentence (-> pad) dropout rate')
    tf.app.flags.DEFINE_float('clip_norm', 0, 'clip gradients to have norm <= this')
    tf.app.flags.DEFINE_integer('batch_size', 128, 'batch size')
    tf.app.flags.DEFINE_integer('lstm_dim', 2048, 'lstm internal dimension')
    tf.app.flags.DEFINE_integer('embed_dim', 50, 'word embedding dimension')
    tf.app.flags.DEFINE_integer('shape_dim', 5, 'shape embedding dimension')
    tf.app.flags.DEFINE_integer('char_dim', 0, 'character embedding dimension')
    tf.app.flags.DEFINE_integer('char_tok_dim', 0, 'character token embedding dimension')
    tf.app.flags.DEFINE_string('char_model', 'lstm', 'character embedding model (lstm, cnn)')

    tf.app.flags.DEFINE_integer('max_finetune_epochs', 100, 'train for this many epochs')
    tf.app.flags.DEFINE_integer('max_context_epochs', 100, 'train for this many epochs')

    tf.app.flags.DEFINE_integer('max_epochs', 100, 'train for this many epochs')
    # tf.app.flags.DEFINE_integer('max_seq_len', 50, 'maximum sequence length')
    tf.app.flags.DEFINE_integer('log_every', 2, 'log status every k steps')
    tf.app.flags.DEFINE_string('embeddings', '', 'file of pretrained embeddings to use')
    tf.app.flags.DEFINE_string('nonlinearity', 'relu', 'nonlinearity function to use (tanh, sigmoid, relu)')
    tf.app.flags.DEFINE_boolean('until_convergence', False, 'whether to run until convergence')
    tf.app.flags.DEFINE_boolean('evaluate_only', False, 'whether to only run evaluation')
    tf.app.flags.DEFINE_string('layers', '', 'json definition of layers (dilation, filters, width)')
    tf.app.flags.DEFINE_boolean('pretrained_pad', True, 'whether to use pretrained pad/oov embeddings or not')
    tf.app.flags.DEFINE_string('print_preds', '', 'print out predictions (for conll eval script) to given file (or do not if empty)')
    tf.app.flags.DEFINE_boolean('viterbi', False, 'whether to use viberbi inference')
    tf.app.flags.DEFINE_string('layers2', '', 'second series of dilations (in place of viterbi)')
    tf.app.flags.DEFINE_boolean('frontend_batch_norm', False, 'whether to perform batch normalization')
    tf.app.flags.DEFINE_boolean('context_batch_norm', False, 'whether to perform batch normalization')
    tf.app.flags.DEFINE_boolean('train_eval', False, 'whether to report train accuracy')
    tf.app.flags.DEFINE_boolean('memmap_train', True, 'whether to load all training examples into memory')
    tf.app.flags.DEFINE_integer('context_residual_layers', 0, 'whether to use residual layers in context')
    tf.app.flags.DEFINE_integer('frontend_residual_layers', 0, 'whether to use residual layers in frontend')
    tf.app.flags.DEFINE_boolean('projection', False, 'whether to do final halving projection (front end)')
    tf.app.flags.DEFINE_boolean('pad_samples_per_example', 1, 'Number of padding samples to take per example')

    tf.app.flags.DEFINE_integer('block_repeats', 1, 'number of times to repeat the stacked dilations block')
    tf.app.flags.DEFINE_boolean('pool_blocks', False, 'whether to max pool block outputs')
    tf.app.flags.DEFINE_boolean('share_repeats', True, 'whether to share parameters between blocks')

    tf.app.flags.DEFINE_boolean('fancy_blocks', False, '')
    tf.app.flags.DEFINE_boolean('residual_blocks', False, '')


    tf.app.flags.DEFINE_string('loss', 'mean', '')
    tf.app.flags.DEFINE_float('margin', 1.0, 'margin')

    tf.app.flags.DEFINE_float('char_input_dropout', 1.0, 'dropout for character embeddings')

    tf.app.flags.DEFINE_float('save_min', 0.0, 'min accuracy before saving')


    tf.app.flags.DEFINE_boolean('update_frontend', False, 'whether to update pretrained front-end')

    tf.app.flags.DEFINE_boolean('start_end', False, 'whether using start/end or just pad between sentences')
    tf.app.flags.DEFINE_boolean('predict_pad', False, 'whether to predict padding labels')
    tf.app.flags.DEFINE_float('regularize_pad_penalty', 0.0, 'penalty for pad regularization')
    tf.app.flags.DEFINE_float('regularize_drop_penalty', 0.0, 'penalty for dropout regularization')
    tf.app.flags.DEFINE_float('context_regularize_drop_penalty', 0.0, 'penalty for dropout regularization')


    tf.app.flags.DEFINE_integer('max_additional_pad', 0, 'max additional padding to add')

    tf.app.flags.DEFINE_boolean('documents', False, 'whether each example is a document (default: sentence)')
    tf.app.flags.DEFINE_boolean('ontonotes', False, 'evaluate each domain of ontonotes seperately')

    tf.app.run()
