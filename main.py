from .utils import *
from .model import *
from .fgsm  import *

def main():
    labels = read_in_labels("labels.txt")
    result = []
    result_fgsm_list = []
    result_fgsm_random = []
    epsilon_0_list = [0.1*i for i in range(1,11)]
    model = Model("param.txt")
    for img_name in range(1,155):
        path = "pgm/"+str(img_name)+".pgm"
        x = Image(path).flattened_vector
        x_fgsmized_list = [fgsm_an_image(x,labels[img_name - 1],epsilon_0,model) for epsilon_0 in epsilon_0_list]
        x_fgsmized_random = fgsm_an_image(x,labels[img_name - 1],0.1,model,for_test=True)
        result.append(model(x)["max_result"])
        result_fgsm_list.append([model(x_fgsmized)["max_result"] for x_fgsmized in x_fgsmized_list])
        result_fgsm_random.append(model(x_fgsmized_random)["max_result"])
    acc_various_models = []
    for i in range(10):
        result_from_single_model = [result_fgsm[i] for result_fgsm in result_fgsm_list]
        acc_various_models.append(compute_accuracy(result_from_single_model,labels))
    acc = {"acc_origin" : compute_accuracy(result,labels),\
    "acc_fgsm" : acc_various_models,\
    "acc_fgsm_random" : compute_accuracy(result_fgsm_random,labels)}
    print(acc)

if __name__ == '__main__':
    main()