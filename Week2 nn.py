import numpy as np
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.utils import to_categorical

#defining ReLu and its derivative for finding the activation values for first hidden layer
def relu(x):
    return np.maximum(0, x)


def d_relu(x):
    return (x > 0).astype(float)


#defining softmax so to get the values of output layer such that the sum of all the values is 1 to get a probabilistic approach
def soft_max(x):
    x_shifted = x - np.max(x, axis=0, keepdims=True) 
    exp_x = np.exp(x_shifted)
    return exp_x / np.sum(exp_x, axis=0, keepdims=True)
    

#getting accuracy for each epoch
def get_accuracy(y_pred, Y):
    pred_labels = np.argmax(y_pred, axis=0)
    true_labels = np.argmax(Y, axis=0)
    return np.mean(pred_labels == true_labels)


#the actual neural network function
def neural_network(X, Y, lr, epochs, input_layer, hidden1, output_layer, batch_size):
    
    n = X.shape[1]  

    #initiallizing weights and biases randomly
    W1 = np.random.randn(hidden1, input_layer) * np.sqrt(2. / input_layer)
    b1 = np.zeros((hidden1, 1))
    W2 = np.random.randn(output_layer, hidden1) * np.sqrt(2. / hidden1)
    b2 = np.zeros((output_layer, 1)) 

    for epoch in range(epochs):

        #shuffling the data using a random permutation
        indices = np.random.permutation(n)
        X_shuffled = X[:, indices]
        Y_shuffled = Y[:, indices]

        #updating the biases and weights in small batches for faster training
        
        for i in range(0, n, batch_size):
            
            X_batch = X_shuffled[:, i:i+batch_size]
            Y_batch = Y_shuffled[:, i:i+batch_size]
            m = X_batch.shape[1]

            
            Z1 = W1 @ X_batch + b1
            A = relu(Z1)
            Z2 = W2 @ A + b2
            Y_pred = soft_max(Z2)

            
            dy = Y_pred - Y_batch
            dw2 = (1 / m) * (dy @ A.T)
            db2 = (1 / m) * np.sum(dy, axis=1, keepdims=True)

            dz1 = (W2.T @ dy) * d_relu(Z1)
            dw1 = (1 / m) * (dz1 @ X_batch.T)
            db1 = (1 / m) * np.sum(dz1, axis=1, keepdims=True)

            
            W1 -= lr * dw1
            b1 -= lr * db1
            W2 -= lr * dw2
            b2 -= lr * db2

       
        Z1 = W1 @ X + b1
        A = relu(Z1)
        Z2 = W2 @ A + b2
        Y_pred = soft_max(Z2)

        #printing accuracy and loss for every 10th epoch
        if (epoch + 1) % 1 == 0:
            acc = get_accuracy(Y_pred, Y)
            print(f"Epoch {epoch+1} | Accuracy: {acc:.4f}")

    return W1, b1, W2, b2


#loading the data
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

X = x_train.reshape(-1, 784) / 255.0
X = X.T
Y = to_categorical(y_train).T

input_layer = 784
hidden1 = 128
output_layer = 10
lr = 0.01
epochs = 20
batch_size = 64  


W1, b1, W2, b2 = neural_network(X, Y, lr, epochs, input_layer, hidden1, output_layer, batch_size)

