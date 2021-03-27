from .utils import *
from .model import *

def main():
    labels = read_in_labels("labels.txt")
    result = []
    model = Model("param.txt")
    for img_name in range(1,155):
        path = "pgm/"+str(img_name)+".pgm"
        x = Image(path).flattened_vector
        result.append(model(x))
    acc = compute_accuracy(result,labels)
    print(acc)

if __name__ == '__main__':
    main()