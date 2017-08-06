import os

from surprise import Dataset
from surprise import Reader
from surprise import SVD


def model_train(rating_dataset=None):
    if rating_dataset is None:
        data = Dataset.load_builtin('ml-100k')
    else:
        # path to dataset file
        file_path = os.path.expanduser(rating_dataset)

        # As we're loading a custom dataset, we need to define a reader. In the
        # movielens-100k dataset, each line has the following format:
        # 'user item rating timestamp', separated by '\t' characters.
        reader = Reader(line_format='user item rating timestamp', sep='\t')

        data = Dataset.load_from_file(file_path, reader=reader)
    # Retrieve the trainset.
    trainset = data.build_full_trainset()
    # Build an algorithm, and train it.
    algo = SVD()
    algo.train(trainset)
    return algo
