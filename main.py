from .utils     import *
from .model     import *
from .fgsm      import *
from .metrics   import *
import sys

def main(argv):
    labels = read_in_labels("labels.txt")
    times_to_repeat_fgsm = int(argv[1])
    # to store the output from result without fgsm
    result = []
    # to store result output from fgsm models with different epsilon0
    result_fgsm_list = []
    # to store result output from random sign
    result_fgsm_random = []
    # experiment with 10 epsilon0 range from 1 to 10
    epsilon_0_list = [0.1*i for i in range(1,11)]
    model = Model("param.txt")
    for img_name in range(1,155):
        path = "pgm/"+str(img_name)+".pgm"
        x = Image(path).flattened_vector
        x_fgsmized_list = [repeat_fgsm(x,labels,img_name,epsilon_0,model,times_to_repeat_fgsm) for epsilon_0 in epsilon_0_list]
        x_fgsmized_random = fgsm_an_image(x,labels[img_name - 1],0.1,model,for_test=True)
        result.append(model(x)["max_result"])
        result_fgsm_list.append([model(x_fgsmized)["max_result"] for x_fgsmized in x_fgsmized_list])
        result_fgsm_random.append(model(x_fgsmized_random)["max_result"])
    # accuracy of mdoels with different epsilon0
    acc_various_models = []
    for i in range(10):
        result_from_single_model = [result_fgsm[i] for result_fgsm in result_fgsm_list]
        acc_various_models.append(compute_accuracy(result_from_single_model,labels))
    acc = {"acc_origin" : compute_accuracy(result,labels),\
    "acc_fgsm" : acc_various_models,\
    "acc_fgsm_random" : compute_accuracy(result_fgsm_random,labels)}
    print(acc)
    draw_line_chart(acc["acc_fgsm"],[0.1*i for i in range(1,11)],"repeat"+str(times_to_repeat_fgsm))

def repeat_fgsm(x,labels,img_name,epsilon_0,model,times):
    for _ in range(times):
        result = fgsm_an_image(x,labels[img_name - 1],epsilon_0,model)
        x = result
    return result

if __name__ == '__main__':
    main(sys.argv)