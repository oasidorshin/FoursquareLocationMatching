import pickle


def pickle_save(obj, filename):
    pickle.dump(obj, open(filename, 'wb'))


def pickle_load(filename):
    return pickle.load(open(filename, 'rb'))


def jaccard_score(target_list, preds_list):
    result = 0.0

    for row_i in range(len(target_list)):
        targets = set(target_list[row_i])
        preds = set(preds_list[row_i])

        result += float(len(targets.intersection(preds)) /
                        len(targets.union(preds)))

    return result / len(target_list)
