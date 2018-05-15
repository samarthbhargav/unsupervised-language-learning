import codecs
import random
import sys


def sentence_iterator(file):
    with codecs.open(file, "r", "utf-8") as reader:
        for line in reader:
            yield line

if __name__ == '__main__':
    if len(sys.argv) != 6:
        print("Error")
        sys.exit(-1)

    source_file_x = sys.argv[1]
    source_file_y = sys.argv[2]
    dest_file_x = sys.argv[3]
    dest_file_y = sys.argv[4]
    n_samples = sys.argv[5]

    random.seed(42)
    # read through the file
    n_lines = sum([1 for _ in sentence_iterator(sys.argv[1])] )
    assert n_lines == sum([1 for _ in sentence_iterator(sys.argv[2])] )


    line_idx = list(range(n_lines))
    lines = set()
    while len(lines) < int(n_samples):
        lines.add(random.choice(line_idx))

    with codecs.open(dest_file_x, "w", "utf-8") as writer_x, codecs.open(dest_file_y, "w", "utf-8") as writer_y:
        for line, (sentence_x, sentence_y) in enumerate(zip(sentence_iterator(source_file_x), sentence_iterator(source_file_y))):
            if line not in lines:
                continue
            writer_x.write(sentence_x)
            writer_y.write(sentence_y)
