import math

class Vector:
    def __init__(self,values):
        # values : a list of element in a vector
        self.length = len(values)   
        self.values = values

    def sum_with_vector(self,y):
        s = [p+q for p,q in zip(self.values,y.values)]
        return Vector(s)
    
    def minus_a_vector(self,y):
        s = [p-q for p,q in zip(self.values,y.values)]
        return Vector(s)

    def relu(self):
        result = []
        for i in self.values:
            if i>0:
                result.append(i)
            else:
                result.append(0)
        return Vector(result)
    
    def softmax(self):
        result = [math.exp(i) for i in self.values]
        s = sum(result)
        result = [i/s for i in result]
        return Vector(result)

class Matrix:
    def __init__(self,vectors):
        # vectors : a list of vector
        # matrix = [v1^T,v2^T,v3^T.....]
        self.rows = vectors
        self.shape = [len(vectors),vectors[0].length]

    def multiply_with_a_vector(self,vector):
        result = []
        for row in self.rows:
            result.append(sum([p*q for p,q in zip(row.values,vector.values)]))
        return Vector(result)
    
    def transpose(self):
        N, M = self.shape
        transposed_matrix = []
        for i in range(M):
            vector = []
            for j in range(N):
                vector.append(self.rows[j].values[i])
            vector = Vector(vector)
            transposed_matrix.append(vector)
        return Matrix(transposed_matrix)
    
class Image:
    def __init__(self,path):
        # Normalized by 255
        self.flattened_vector = Vector(self.read_pgm(path))

    def read_pgm(self,path):
        with open(path) as f:
            lines = f.readlines()
        for l in list(lines):
            if l[0] == '#':
                lines.remove(l)
        # here,it makes sure it is ASCII format (P2)
        assert lines[0].strip() == 'P2' 
        # Converts data to a list of integers
        data = []
        for line in lines[1:]:
            data.extend([int(c)/255 for c in line.split()])
        return data[3:]

class Parameters:
    def __init__(self,path):
        self.H = 256
        self.C = 23
        self.N = 1024
        # w is a matrix, b is a vector
        self.w1,self.b1,self.w2,self.b2,self.w3,self.b3=self.read_in_params(path)

    def read_in_params(self,path):
        target = []
        # Read the number of lines according to the list
        num_lines = [self.H,1,self.H,1,self.C,1]
        with open(path) as f:
            lines = f.readlines()
            last = 0 #last time read to
            for i in range(6):
                start = last
                end = sum(num_lines[:i+1])
                lines_slice = lines[start:end]
                result = []
                for line in lines_slice:
                    line = line.strip().split(" ")
                    line = [float(i) for i in line]
                    line = Vector(line)
                    result.append(line)
                if i%2 == 0:
                    # w
                    target.append(Matrix(result))
                else:
                    # b has only 1 line
                    target.append(line)
                last = end
        # target 0-5 are w1,b1,w2,b2,w3,b3 respectively.
        return target[0],target[1],target[2],target[3],target[4],target[5]

def unit_test():
    # test for Vector and Matrix 
    x1 = Vector([1,2,-1])
    x2 = Vector([4,5,6])
    y = Matrix([x1,x2])
    y1 = x1.sum_with_vector(x2)
    y2 = y.multiply_with_a_vector(x1)
    assert y.shape == [2,3]
    assert y1.length == 3
    assert y2.length == 3
    # test for Image
    image = Image("pgm/1.pgm")
    assert image.flattened_vector.length == 1024
    # test for Parameters
    w1 = Parameters("param.txt").w1
    assert w1.shape == [256, 1024]

def argmax(a):
    return max(range(len(a)), key=lambda x: a[x])   

def read_in_labels(path):
        labels = []
        with open(path) as f:
            lines = f.readlines()
            labels = []
            for line in lines:
                line = line.strip() 
                labels.append(int(line))
        return labels

def compute_accuracy(result,labels):
    right = 0
    for x,y in zip(result,labels):
        if x == y:
            right+=1
    return right/len(labels)

if __name__ == '__main__':
    unit_test()