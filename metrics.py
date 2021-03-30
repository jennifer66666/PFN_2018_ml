import numpy as np
import matplotlib.pyplot as plt

def draw_line_chart(acc,epsilon_0):
    x = epsilon_0
    y = acc
    l=plt.plot(x,y)
    plt.title('acc-epsilon0')
    plt.xlabel('epsilon0')
    plt.ylabel('acc')
    plt.show()
    plt.save("acc-epsilon.png")

if __name__ == '__main__':
    # record the output of 10 epsilon0 experiments here
    acc = [0.8181818181818182, 0.7402597402597403, 0.6493506493506493, 0.5909090909090909, \
           0.5324675324675324, 0.512987012987013, 0.45454545454545453, 0.44805194805194803, \
           0.42857142857142855, 0.42207792207792205]
    epsilon_0 = [0.1*i for i in range(1,11)]
    draw_line_chart(acc,epsilon_0)
