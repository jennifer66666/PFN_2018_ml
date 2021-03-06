# PFN_2018_ml
Self-Test for Preferred Network Internship 2018.<br>
The task description, model weights, and dataset can be found [here](https://github.com/pfnet/intern-coding-tasks/tree/master/2018/ml).
## Overall usage
```
# download files needed
git clone https://github.com/pfnet/intern-coding-tasks.git
cp -r intern-coding-tasks/2018/ml ~/ml
cd ~/ml
git clone https://github.com/jennifer66666/PFN_2018_ml.git src

# run main file to print acc in diffrent cases
cd ~/ml
python3 -m src.main

# repeat fgsm for designated times, say, 4 times
python3 -m src.main 4
```
## Part1
The principle for my datatype design is introduced later [here](#appendix-1).
## Part2 
Accuracy_original of the output from feeding in the original data: **0.83766**
## Part3 
  Expriments on FGSM with epsilon0 range from 0.1 to 1 (step = 0.1) show the accuracy_various_models in Figure.1. Notice that the larger
  epsilon0, the better FGSM performs. Although there is no further research on epsilon0 larger than 1, I guess the proportional relation will keep.
<p align="center">
  <img src="https://github.com/jennifer66666/PFN_2018_ml/blob/master/acc_epsilon0.png" width="600" height="400" alt="Figure.1."/>
</p>
Substitue sign(dL_x) with random +1 or -1, accuracy_random decreases a bit from Accuracy_original: **0.83157**.
Random sign model is tried only once here to just verify that FGSM works. Accuracy_random should be different every try, but always less then accuracy_original and larger than accuracy_various_models.

## Part4

### 4.1 Apply FGSM multiple times
By applying FGSM on images multiple times with different epsilon0, we see that for small epsilon0, the result is more affected by the repeat times. However, when epsilon0 proceeding 1, the results almost don't change when repeat times increase.
<p align="center">
  <img src="https://github.com/jennifer66666/PFN_2018_ml/blob/master/acc-epsilon-repeat2.png" width="240" height="160" alt="Figure.2."/><img src="https://github.com/jennifer66666/PFN_2018_ml/blob/master/acc-epsilon-repeat3.png" width="240" height="160" alt="Figure.3."/><img src="https://github.com/jennifer66666/PFN_2018_ml/blob/master/acc-epsilon-repeat4.png" width="240" height="160" alt="Figure.4."/><img src="https://github.com/jennifer66666/PFN_2018_ml/blob/master/acc-epsilon-repeat5.png" width="240" height="160" alt="Figure.5."/><img src="https://github.com/jennifer66666/PFN_2018_ml/blob/master/acc-epsilon-repeat6.png" width="240" height="160" alt="Figure.6."/><img src="https://github.com/jennifer66666/PFN_2018_ml/blob/master/acc-epsilon-repeat7.png" width="240" height="160" alt="Figure.7."/>
</p>

Note that the x axis is epsilon0 range from 0.1 to 1 with step 0.1. The y axis is the accuracy. Different figure titles indicate the times FGSM is repeated on a single image.
## Appendix 1
1. Vector is made over a list of values. Objectivize it by Vector(list_of_values).
2. Matrix is made over a list of vectors. When there is only one vector, the matrix degenerate to a vector. Objectivize it by Matrix(list_of_vectors).
3. The fundamental datatypes within Vector and Matrix are both list. So when implement matrix product with a vector, namely Ax, we default that x is column vector and vector in A is row vector. In other words, I don't transpose a vector when put it into a matrix here, although mathematically I should.
4. Method of an object from Matrix or Vector return a new object of the result, rather than change the obeject.
5. To do calculus in {vector+/-vector, softmax_vector, matrix_multiply_with_vector, matrix_transpose}, we simply call the method of the object.
6. To do calculus outside the range mentioned in 5., we access the content (values/rows) in an object by using vector.values, or matrix.rows.values.
7. Vector after calculus of its methods is still Vector.
8. Matrix transpose to a Matrix, and multiply with a Vector to be a Vector.

