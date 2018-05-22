import numpy as np

# Number of inputs
n = 10

# Threshold activation function
def f(v):
    return 1*(v>0)

# Weight vector 'w' {-1,1}^n, and at least one element is '1'
w = 2 * np.concatenate((np.array([1]), np.random.randint(2,size=(n-1,)))) - 1
print("Weight vector 'w'           :\t" + str(w))

# Perfect input vector 'x_hat'
x_hat = np.maximum(0,w)
print("Perfect input vector 'x_hat':\t" + str(x_hat))

# Activation output 'o_hat' is one (1), meaning it is fully active
o_hat = f(np.dot(w,x_hat))
print("Activation output 'o_hat'   :\t" + str(o_hat))

# Backpropagating activation output 'o_hat' we retrieve 'x_hat'
x_hat_backprop = f(np.dot(np.transpose(w),o_hat))
print("Backpropagating 'o_hat'     :\t" + str(x_hat_backprop))

print("\n")
print("x_hat == x_hat_backprop?    :\t" + str(np.all(x_hat == x_hat_backprop)))
