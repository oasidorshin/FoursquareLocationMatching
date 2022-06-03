import pickle


def pickle_save(obj, filename):
    pickle.dump(obj, open(filename, 'wb'))


def pickle_load(filename):
    return pickle.load(open(filename, 'rb'))
