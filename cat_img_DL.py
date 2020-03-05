import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy
import imageio
from PIL import Image
from scipy import ndimage
from lr_utils import load_dataset


#loading the data (cat/non-cat) 
train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()

m_train = train_set_x_orig.shape[0]
m_test = test_set_x_orig.shape[0]
num_px = train_set_x_orig.shape[1]


#reshape the trainning and test examples
train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0],-1).T
test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0],-1).T

#standardize dataset
train_set_x = train_set_x_flatten/255
test_set_x = test_set_x_flatten/255

#helper function 
def sigmoid(z):
    s = 1/(1+np.exp(-z))
    return s

#initialize parameters
def initialize_with_zeros(dim):
    w = np.zeros((dim,1))
    b = 0
    assert(w.shape==(dim,1))
    assert(isinstance(b,float) or isinstance(b,int))
    return w, b

# forward and backward propagation
def propagation(w,b,X,Y):
    # forward propagation
    m = X.shape[1]
    A = sigmoid(np.dot(w.T,X)+b)
    cost = (-1/m)*(np.dot(Y,np.log(A).T)+np.dot(1-Y,np.log(1-A).T))
    # backward propagation
    dw = (1/m)*np.dot(X, (A-Y).T)
    db = (1/m)*np.sum(A-Y,1)[0]
    cost = np.squeeze(cost)
    grads = {"dw": dw, "db": db}
    return grads, cost

def optimize(w,b,X,Y, num_iterations, learning_rate, print_cost =True):
    costs = []
    for i in range (num_iterations):
        grads, cost = propagation(w,b,X,Y)
        dw = grads["dw"]
        db = grads["db"]
        w=w- learning_rate*dw
        b=b- learning_rate*db
        params = {"w":w, "b":b}
        grads = {"dw":dw, "db":db}
    return params, grads, costs

def predict(w, b, X):
    m =X.shape[1]
    Y_prediction = np.zeros((1,m))
    w = w.reshape(X.shape[0],1)

    A = sigmoid(np.dot(w.T,X)+b)
    print(A)
    for i in range (A.shape[1]):
        if A[0,i] > 0.5:
            Y_prediction[0,i]=1
        else: 
            Y_prediction[0,i]=0
    return Y_prediction
def model(X_train, Y_train, X_test, Y_test, num_iterations = 2000, learning_rate = 0.5, print_cost = True):
    w,b= initialize_with_zeros(X_train.shape[0])
    parameters, grads, costs = optimize(w,b,X_train,Y_train, num_iterations, learning_rate, print_cost)

    w = parameters["w"]
    b = parameters["b"]

    Y_prediction_test = predict(w,b,X_test)
    Y_prediction_train = predict(w,b,X_train)
    d = {"costs": costs,
         "Y_prediction_test": Y_prediction_test,
         "Y_prediction_train" : Y_prediction_train,
         "w" : w,
         "b" : b,
         "learning_rate" : learning_rate,
         "num_iterations": num_iterations}

    return d

d = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations = 2000, learning_rate = 0.005, print_cost=True)
## START CODE HERE ## (PUT YOUR IMAGE NAME) 
while True:
    
    my_image = input("Please enter name of the file or q to quit app:")
    # We preprocess the image to fit your algorithm.
    if my_image == 'q':
        break
    fname = "images/" + my_image
    image = np.array(imageio.imread(fname, as_gray=False))
    image = image/255
    print(image.shape)
    my_image = np.array(Image.fromarray(image.astype(np.uint8)).resize((num_px,num_px))).reshape((1,num_px*num_px*3)).T
    my_predicted_image = predict(d["w"], d["b"], my_image)

    plt.imshow(image)
    print("y = " + str(np.squeeze(my_predicted_image)) + ", your algorithm predicts a \"" + classes[int(np.squeeze(my_predicted_image)),].decode("utf-8") +  "\" picture.")


