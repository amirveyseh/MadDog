#!/usr/bin/env python

"""
Score the predictions with gold labels, using precision, recall and F1 metrics.
"""

import argparse
import sys
from collections import Counter
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix
from collections import defaultdict

from prototype.acronym.zeroshot.model.utils import constant

NO_RELATION = constant.NEGATIVE_LABEL


def parse_arguments():
    parser = argparse.ArgumentParser(description='Score a prediction file using the gold labels.')
    parser.add_argument('gold_file', help='The gold relation file; one relation per line')
    parser.add_argument('pred_file',
                        help='A prediction file; one relation per line, in the same order as the gold file.')
    args = parser.parse_args()
    return args


def score_label_level(key, prediction, verbose=False, verbose_output=False, method='micro'):
    key = [[l] for k in key for l in k]
    prediction = [[l] for k in prediction for l in k]

    org_gold = key.copy()
    org_predictioins = prediction.copy()

    the_key = []
    the_prediction = []

    for i, k in enumerate(key):
        the_key.extend(k)
        the_prediction.extend(prediction[i])

    key = the_key
    prediction = the_prediction

    correct_by_relation = Counter()
    guessed_by_relation = Counter()
    gold_by_relation = Counter()

    # Loop over the data to compute a score
    for row in range(len(key)):
        gold = key[row]
        guess = prediction[row]

        if gold == NO_RELATION and guess == NO_RELATION:
            pass
        elif gold == NO_RELATION and guess != NO_RELATION:
            guessed_by_relation[guess] += 1
        elif gold != NO_RELATION and guess == NO_RELATION:
            gold_by_relation[gold] += 1
        elif gold != NO_RELATION and guess != NO_RELATION:
            guessed_by_relation[guess] += 1
            gold_by_relation[gold] += 1
            if gold == guess:
                correct_by_relation[guess] += 1

    precesions = {}
    recalls = {}
    f1s = {}

    # Print verbose information
    if verbose:
        print("Per-relation statistics:")
        relations = gold_by_relation.keys()
        longest_relation = 0
        for relation in sorted(relations):
            longest_relation = max(len(relation), longest_relation)
        for relation in sorted(relations):
            # (compute the score)
            correct = correct_by_relation[relation]
            guessed = guessed_by_relation[relation]
            gold = gold_by_relation[relation]
            prec = 1.0
            if guessed > 0:
                prec = float(correct) / float(guessed)
            recall = 0.0
            if gold > 0:
                recall = float(correct) / float(gold)
            f1 = 0.0
            if prec + recall > 0:
                f1 = 2.0 * prec * recall / (prec + recall)
            # (print the score)
            sys.stdout.write(("{:<" + str(longest_relation) + "}").format(relation))
            sys.stdout.write("  P: ")
            if prec < 0.1: sys.stdout.write(' ')
            if prec < 1.0: sys.stdout.write(' ')
            sys.stdout.write("{:.2%}".format(prec))
            sys.stdout.write("  R: ")
            if recall < 0.1: sys.stdout.write(' ')
            if recall < 1.0: sys.stdout.write(' ')
            sys.stdout.write("{:.2%}".format(recall))
            sys.stdout.write("  F1: ")
            if f1 < 0.1: sys.stdout.write(' ')
            if f1 < 1.0: sys.stdout.write(' ')
            sys.stdout.write("{:.2%}".format(f1))
            sys.stdout.write("  #: %d" % gold)
            sys.stdout.write("\n")

            precesions[relation] = prec
            recalls[relation] = recall
            f1s[relation] = f1

        print("")

        print('Macro scroes: ')
        print('Macro P: ', sum(precesions.values()) / len(precesions))
        print('Macro R: ', sum(recalls.values()) / len(recalls))
        print('Macro F1: ', sum(f1s.values()) / len(f1s))

    # Print the aggregate score
    if verbose:
        print("Final Score:")
    prec_micro = 1.0
    if sum(guessed_by_relation.values()) > 0:
        prec_micro = float(sum(correct_by_relation.values())) / float(sum(guessed_by_relation.values()))
    recall_micro = 0.0
    if sum(gold_by_relation.values()) > 0:
        recall_micro = float(sum(correct_by_relation.values())) / float(sum(gold_by_relation.values()))
    f1_micro = 0.0
    if prec_micro + recall_micro > 0.0:
        f1_micro = 2.0 * prec_micro * recall_micro / (prec_micro + recall_micro)
    print("Precision (micro): {:.3%}".format(prec_micro))
    print("   Recall (micro): {:.3%}".format(recall_micro))
    print("       F1 (micro): {:.3%}".format(f1_micro))

    relations = ['B-Term', 'I-Term', 'B-Definition', 'I-Definition', 'B-Qualifier', 'I-Qualifier']

    if verbose_output:
        for relation in relations:
            if relation in precesions:
                print('{:.2f}'.format(precesions[relation] * 100))
            else:
                print('-')
        print('############################################################')
        for relation in relations:
            if relation in precesions:
                print('{:.2f}'.format(recalls[relation] * 100))
            else:
                print('-')
        print('############################################################')
        for relation in relations:
            if relation in precesions:
                print('{:.2f}'.format(f1s[relation] * 100))
            else:
                print('-')

    if method == 'macro':
        macro_f1 = f1_score(org_gold, org_predictioins, average='macro')
        macro_p = precision_score(org_gold, org_predictioins, average='macro')
        macro_r = recall_score(org_gold, org_predictioins, average='macro')
        return macro_p, macro_r, macro_f1
    else:
        return prec_micro, recall_micro, f1_micro


def score_expansion(key, prediction, verbos=False, method='macro'):
    correct = 0
    for i in range(len(key)):
        if key[i] == prediction[i]:
            correct += 1
    acc = correct / len(prediction)

    expansions = set()

    correct_per_expansion = defaultdict(int)
    total_per_expansion = defaultdict(int)
    pred_per_expansion = defaultdict(int)
    for i in range(len(key)):
        expansions.add(key[i])
        total_per_expansion[key[i]] += 1
        pred_per_expansion[prediction[i]] += 1
        if key[i] == prediction[i]:
            correct_per_expansion[key[i]] += 1

    precs = defaultdict(int)
    recalls = defaultdict(int)

    for exp in expansions:
        precs[exp] = correct_per_expansion[exp] / pred_per_expansion[exp] if exp in pred_per_expansion else 1
        recalls[exp] = correct_per_expansion[exp] / total_per_expansion[exp]

    micro_prec = sum(correct_per_expansion.values()) / sum(pred_per_expansion.values())
    micro_recall = sum(correct_per_expansion.values()) / sum(total_per_expansion.values())
    micro_f1 = 2 * micro_prec * micro_recall / (micro_prec + micro_recall)

    macro_prec = sum(precs.values()) / len(precs)
    macro_recall = sum(recalls.values()) / len(recalls)
    macro_f1 = 2 * macro_prec * macro_recall / (macro_prec + macro_recall)

    if verbos:
        print('Accuracy: {:.3%}'.format(acc))
        print('-' * 10)
        print('Micro Precision: {:.3%}'.format(micro_prec))
        print('Micro Recall: {:.3%}'.format(micro_recall))
        print('Micro F1: {:.3%}'.format(micro_f1))
        print('-' * 10)
        print('Macro Precision: {:.3%}'.format(macro_prec))
        print('Macro Recall: {:.3%}'.format(macro_recall))
        print('Macro F1: {:.3%}'.format(macro_f1))
        print('-' * 10)
        # print('Scores from sklearn: ')
        # print(f1_score(key, prediction, average='macro'))
        # print(precision_score(key, prediction, average='macro'))
        # print(recall_score(key, prediction, average='macro'))
        # print(total_per_expansion)
        # print(pred_per_expansion)
        # print(correct_per_expansion)
        # print(recalls)

    if method == 'micro':
        return micro_prec, micro_recall, micro_f1
    elif method == 'macro':
        return macro_prec, macro_recall, macro_f1


def score_phrase_level(key, predictions, verbos=False, method='macro'):
    gold_shorts = set()
    gold_longs = set()
    pred_shorts = set()
    pred_longs = set()

    def find_phrase(seq, shorts, longs):

        for i, sent in enumerate(seq):
            short_phrase = []
            long_phrase = []
            for j, w in enumerate(sent):
                if 'B' in w or 'O' in w:
                    if len(long_phrase) > 0:
                        longs.add(str(i) + '-' + str(long_phrase[0]) + '-' + str(long_phrase[-1]))
                        long_phrase = []
                    if len(short_phrase) > 0:
                        shorts.add(str(i) + '-' + str(short_phrase[0]) + '-' + str(short_phrase[-1]))
                        short_phrase = []
                if 'short' in w:
                    short_phrase.append(j)
                if 'long' in w:
                    long_phrase.append(j)
            if len(long_phrase) > 0:
                longs.add(str(i) + '-' + str(long_phrase[0]) + '-' + str(long_phrase[-1]))
            if len(short_phrase) > 0:
                shorts.add(str(i) + '-' + str(short_phrase[0]) + '-' + str(short_phrase[-1]))

    find_phrase(key, gold_shorts, gold_longs)
    find_phrase(predictions, pred_shorts, pred_longs)

    # print(len(pred_shorts))
    # print(len(gold_shorts))
    # for sf in pred_shorts:
    #     if sf not in gold_shorts:
    #         print(sf)
    #         ind = int(sf.split('-')[0])
    #         print(list(zip(range(len(predictions[ind])), predictions[ind])))
    #         print(predictions[ind])
    #         print(list(zip(range(len(key[ind])), key[ind])))
    #         print(key[ind])
    #         exit(1)

    def find_prec_recall_f1(pred, gold):
        correct = 0
        for phrase in pred:
            if phrase in gold:
                correct += 1
        # print(correct)
        prec = correct / len(pred) if len(pred) > 0 else 1
        recall = correct / len(gold) if len(gold) > 0 else 1
        f1 = 2 * prec * recall / (prec + recall) if prec + recall > 0 else 0
        return prec, recall, f1

    prec_short, recall_short, f1_short = find_prec_recall_f1(pred_shorts, gold_shorts)
    prec_long, recall_long, f1_long = find_prec_recall_f1(pred_longs, gold_longs)
    precision_micro, recall_micro, f1_micro = find_prec_recall_f1(pred_shorts.union(pred_longs),
                                                                  gold_shorts.union(gold_longs))

    precision_macro = (prec_short + prec_long) / 2
    recall_macro = (recall_short + recall_long) / 2
    f1_macro = 2 * precision_macro * recall_macro / (
                precision_macro + recall_macro) if precision_macro + recall_macro > 0 else 0

    if verbos:
        print('Shorts: P: {:.2%}, R: {:.2%}, F1: {:.2%}'.format(prec_short, recall_short, f1_short))
        print('Longs: P: {:.2%}, R: {:.2%}, F1: {:.2%}'.format(prec_long, recall_long, f1_long))
        print('micro scores: P: {:.2%}, R: {:.2%}, F1: {:.2%}'.format(precision_micro, recall_micro, f1_micro))
        print('macro scores: P: {:.2%}, R: {:.2%}, F1: {:.2%}'.format(precision_macro, recall_macro, f1_macro))

    if method == 'macro':
        return precision_macro, recall_macro, f1_macro
    else:
        return precision_micro, recall_micro, f1_micro


def score(key, prediction, verbose=False, verbose_output=False, method='macro', level='phrase'):
    if level == 'label':
        return score_label_level(key, prediction, verbose, verbose_output, method)
    else:
        return score_phrase_level(key, prediction, verbose, method)


if __name__ == "__main__":
    # Parse the arguments from stdin
    args = parse_arguments()
    key = [str(line).rstrip('\n') for line in open(str(args.gold_file))]
    prediction = [str(line).rstrip('\n') for line in open(str(args.pred_file))]

    # Check that the lengths match
    if len(prediction) != len(key):
        print("Gold and prediction file must have same number of elements: %d in gold vs %d in prediction" % (
        len(key), len(prediction)))
        exit(1)

    # Score the predictions
    score(key, prediction, verbose=True)

