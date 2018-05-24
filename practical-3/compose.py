import numpy as np


def compose(embeddings, method):
    return {
        "avg": np.mean(embeddings, axis=0),
        "max": np.max(embeddings, axis=0),
        "min": np.min(embeddings, axis=0),
        "concat": np.hstack((np.mean(embeddings, axis=0),
                             np.max(embeddings, axis=0),
                             np.min(embeddings, axis=0)))
    }[method]


if __name__ == '__main__':
    a = [
        [1, 2],
        [3, 4]
    ]
    print(compose(a, "concat"))
