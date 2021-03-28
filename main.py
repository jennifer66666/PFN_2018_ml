from .utils import *
from .model import *
from .fgsm  import *

def main():
    labels = read_in_labels("labels.txt")
    result = []
    result_fgsm = []
    result_fgsm_random = []
    model = Model("param.txt")
    for img_name in range(1,155):
        path = "pgm/"+str(img_name)+".pgm"
        x = Image(path).flattened_vector
        x_fgsmized = fgsm_an_image(x,labels[img_name - 1],0.1,model)
        x_fgsmized_random = fgsm_an_image(x,labels[img_name - 1],0.1,model,for_test=True)
        result.append(model(x)["max_result"])
        result_fgsm.append(model(x_fgsmized)["max_result"])
        result_fgsm_random.append(model(x_fgsmized_random)["max_result"])
    acc = {"acc_origin" : compute_accuracy(result,labels),\
    "acc_fgsm" : compute_accuracy(result_fgsm,labels),\
    "acc_fgsm_random" : compute_accuracy(result_fgsm_random,labels)}
    print(acc)

if __name__ == '__main__':
    main()