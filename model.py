from .utils import *

class Model:
    def __init__(self,path):
        self.params = Parameters(path)
    
    def __call__(self,x):
        a1 = self.params.w1.multiply_with_a_vector(x).sum_with_vector(self.params.b1)
        h1 = a1.relu()      
        a2 = self.params.w2.multiply_with_a_vector(h1).sum_with_vector(self.params.b2)
        h2 = a2.relu()
        y = self.params.w3.multiply_with_a_vector(h2).sum_with_vector(self.params.b3)
        result = y.softmax()
        max_result = argmax(result.values)+1
        return {"max_result":max_result,"a1":a1,"a2":a2,"result":result}

        