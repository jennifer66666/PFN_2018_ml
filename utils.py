import math

class Vector:
    def __init__(self,values):
        # values : a list of element in a vector
        self.length = len(values)   
        self.values = values

    def sum_with_vector(self,y):
        s = [p+q for p,q in zip(self.values,y.values)]
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
        result = [math.exp(i) for i in range(self.length)]
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
            

def main():
    x1 = Vector([1,2,-1])
    x2 = Vector([4,5,6])
    y = Matrix([x1,x2])
    y2 = x1.sum_with_vector(x2)
    y3 = y.multiply_with_a_vector(x1)
    """     for i in y.rows:
        for j in i.values:
            print(j)
    for i in y.transpose().rows:
        for j in i.values:
            print(j) """

    """     for i in y3.values:
        print(i) """
    for i in x1.softmax().values:
        print(i)

if __name__ == '__main__':
    main()