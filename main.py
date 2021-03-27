from .utils import *
from .model import *
import os

def main():
    imgs_name = os.listdir("pgm")
    labels = read_in_labels("labels.txt")
    result = []
    for img_name in imgs_name:
        x = Image("pgm/"+img_name).flatten()
        model = Model("param.txt")
        result.append(model(x))
    acc = compute_accuracy(result,labels)
    
    print(acc)

        


if __name__ == '__main__':
    main()