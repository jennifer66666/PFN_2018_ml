
from .model import *
from .utils import *
import random

def fgsm_an_image(x,label,epsilon_0,f,for_test = False):
    # x is a flattened and normalized image.
    dL_x = compute_dL_x(label,x,f)
    if not for_test:
        epsilon = compute_epsilon(epsilon_0,dL_x)
    else: 
        # set random sign to show baseline
        epsilon = compute_epsilon(epsilon_0,dL_x,for_test = for_test)
    return x.sum_with_vector(epsilon)

def compute_dL_x(label,x,f):
    # f is a Model object
    model_output = f(x)
    delta_t = [0] * x.length
    delta_t[label] = 1
    delta_t = Vector(delta_t)
    dL_y  = model_output["result"].minus_a_vector(delta_t)
    dL_h2 = f.params.w3.transpose().multiply_with_a_vector(dL_y)
    dL_a2 = backward(dL_h2,model_output["a2"])
    dL_h1 = f.params.w2.transpose().multiply_with_a_vector(dL_a2)
    dL_a1 = backward(dL_h1,model_output["a1"])
    dL_x  = f.params.w1.transpose().multiply_with_a_vector(dL_a1)
    return dL_x

def compute_epsilon(epsilon_0,dL_x,for_test=False):
    if not for_test:
        signed_dL_x = sign(dL_x)
    else:
        signed_dL_x = random_sign(dL_x.length)
    result = []
    for i in signed_dL_x.values:
        result.append(epsilon_0 * i)
    return Vector(result)

def random_sign(length):
    result = []
    for i in range(length):
        result += [ 1 if random.random() < 0.5 else -1 ]
    return Vector(result)

def sign(vector):
    result = []
    for i in vector.values:
        if i>0:
            result.append(1)
        else:
            result.append(-1)
    return Vector(result)

def backward(vector_p,vector_q):
    result = []
    for p,q in zip(vector_p.values,vector_q.values):
        if q > 0:
            result.append(p)
        else:
            result.append(0)
    return Vector(result)



    