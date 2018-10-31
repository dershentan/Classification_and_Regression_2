import numpy as np
from scipy.io import loadmat
from scipy.optimize import minimize
from sklearn.svm import SVC
import pickle

def preprocess():
    """ 
     Input:
     Although this function doesn't have any input, you are required to load
     the MNIST data set from file 'mnist_all.mat'.

     Output:
     train_data: matrix of training set. Each row of train_data contains 
       feature vector of a image
     train_label: vector of label corresponding to each image in the training
       set
     validation_data: matrix of training set. Each row of validation_data 
       contains feature vector of a image
     validation_label: vector of label corresponding to each image in the 
       training set
     test_data: matrix of training set. Each row of test_data contains 
       feature vector of a image
     test_label: vector of label corresponding to each image in the testing
       set
    """

    mat = loadmat('mnist_all.mat')  # loads the MAT object as a Dictionary

    n_feature = mat.get("train1").shape[1]
    n_sample = 0
    for i in range(10):
        n_sample = n_sample + mat.get("train" + str(i)).shape[0]
    n_validation = 1000
    n_train = n_sample - 10 * n_validation

    # Construct validation data
    validation_data = np.zeros((10 * n_validation, n_feature))
    for i in range(10):
        validation_data[i * n_validation:(i + 1) * n_validation, :] = mat.get("train" + str(i))[0:n_validation, :]

    # Construct validation label
    validation_label = np.ones((10 * n_validation, 1))
    for i in range(10):
        validation_label[i * n_validation:(i + 1) * n_validation, :] = i * np.ones((n_validation, 1))

    # Construct training data and label
    train_data = np.zeros((n_train, n_feature))
    train_label = np.zeros((n_train, 1))
    temp = 0
    for i in range(10):
        size_i = mat.get("train" + str(i)).shape[0]
        train_data[temp:temp + size_i - n_validation, :] = mat.get("train" + str(i))[n_validation:size_i, :]
        train_label[temp:temp + size_i - n_validation, :] = i * np.ones((size_i - n_validation, 1))
        temp = temp + size_i - n_validation

    # Construct test data and label
    n_test = 0
    for i in range(10):
        n_test = n_test + mat.get("test" + str(i)).shape[0]
    test_data = np.zeros((n_test, n_feature))
    test_label = np.zeros((n_test, 1))
    temp = 0
    for i in range(10):
        size_i = mat.get("test" + str(i)).shape[0]
        test_data[temp:temp + size_i, :] = mat.get("test" + str(i))
        test_label[temp:temp + size_i, :] = i * np.ones((size_i, 1))
        temp = temp + size_i

    # Delete features which don't provide any useful information for classifiers
    sigma = np.std(train_data, axis=0)
    index = np.array([])
    for i in range(n_feature):
        if (sigma[i] > 0.001):
            index = np.append(index, [i])
    train_data = train_data[:, index.astype(int)]
    validation_data = validation_data[:, index.astype(int)]
    test_data = test_data[:, index.astype(int)]

    # Scale data to 0 and 1
    train_data /= 255.0
    validation_data /= 255.0
    test_data /= 255.0

    return train_data, train_label, validation_data, validation_label, test_data, test_label


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def blrObjFunction(initialWeights, *args):
    """
    blrObjFunction computes 2-class Logistic Regression error function and
    its gradient.

    Input:
        initialWeights: the weight vector (w_k) of size (D + 1) x 1 
        train_data: the data matrix of size N x D
        labeli: the label vector (y_k) of size N x 1 where each entry can be either 0 or 1 representing the label of corresponding feature vector

    Output: 
        error: the scalar value of error function of 2-class logistic regression
        error_grad: the vector of size (D+1) x 1 representing the gradient of
                    error function
    """
    train_data, labeli = args
    # print(train_data.shape)
    n_data = train_data.shape[0] # N
    n_features = train_data.shape[1] # D
    error = 0
    error_grad = np.zeros((n_features + 1, 1))

    ##################
    # YOUR CODE HERE #
    ##################
    # HINT: Do not forget to add the bias term to your input data

    # Adding Bias to front of train_data
    train_data = np.append(np.ones((n_data,1)),train_data,axis = 1)
    # print("append ",train_data.shape)
    # Calculating theta (shape N x 1)
    theta = sigmoid(np.matmul(train_data,initialWeights))
    theta = theta.reshape((n_data,1))
    # Calculating error based on equation 2
    temp1 = np.sum(np.matmul(labeli.transpose(),np.log(theta)))
    temp2 = np.sum(np.matmul((1-labeli).transpose(),np.log(1-theta)))
    error = -(np.sum(temp1+temp2))/n_data
    # Calculating gradient
    error_grad =1/n_data * np.matmul (train_data.transpose(),theta - labeli)
    # print(error,error_grad.shape)
    return error, error_grad.flatten()


def blrPredict(W, data):
    """
     blrObjFunction predicts the label of data given the data and parameter W 
     of Logistic Regression

     Input:
         W: the matrix of weight of size (D + 1) x 10. Each column is the weight
         vector of a Logistic Regression classifier.
         X: the data matrix of size N x D

     Output:
         label: vector of size N x 1 representing the predicted label of
         corresponding feature vector given in data matrix

    """
    label = np.zeros((data.shape[0], 1))

    ##################
    # YOUR CODE HERE #
    ##################
    # HINT: Do not forget to add the bias term to your input data

    # appending bais
    data_wb = np.append(np.ones(shape=(data.shape[0], 1)), data, axis=1)

    # calculating the output and getting the label
    output = sigmoid(np.dot(data_wb, W))
    label = output.argmax(axis=1).reshape(data_wb.shape[0], 1)

    return label


def mlrObjFunction(params, *args):
    """
    mlrObjFunction computes multi-class Logistic Regression error function and
    its gradient.

    Input:
        params: the weight vector of size (D + 1) x K
        train_data: the data matrix of size N x D
        labeli: the label vector of size N x K where each entry can be either 0 or 1
                representing the label of corresponding feature vector

    Output:
        error: the scalar value of error function of multi-class logistic regression
        error_grad: the vector of size (D+1) x 10 representing the gradient of
                    error function
    """
    train_data, labeli = args
    n_data = train_data.shape[0]
    n_feature = train_data.shape[1]
    n_class = labeli.shape[1]
    error = 0
    error_grad = np.zeros((n_feature + 1, n_class))

    # Reshape the weights
    params = params.reshape((n_feature+1,n_class))
    ##################
    # YOUR CODE HERE #
    ##################
    # HINT: Do not forget to add the bias term to your input data
    train_data = np.append(np.ones((n_data, 1)), train_data, axis=1)
    # Calculate theta N  x K
    theta = np.exp(np.matmul(train_data,params))
    theta = theta/(np.sum(theta,axis = 1).reshape((n_data,1)))
    # calculate negative loglikelihood
    error = -np.sum(np.multiply(labeli,np.log(theta)))

    # Calculate gradient
    error_grad = np.matmul(train_data.transpose(), theta - labeli).flatten()
    # print(np.sum(error_grad))
    return error, error_grad


def mlrPredict(W, data):
    """
     mlrObjFunction predicts the label of data given the data and parameter W
     of Logistic Regression

     Input:
         W: the matrix of weight of size (D + 1) x 10. Each column is the weight
         vector of a Logistic Regression classifier.
         X: the data matrix of size N x D

     Output:
         label: vector of size N x 1 representing the predicted label of
         corresponding feature vector given in data matrix

    """
    label = np.zeros((data.shape[0], 1))

    ##################
    # YOUR CODE HERE #
    ##################
    # HINT: Do not forget to add the bias term to your input data
    data = np.append(np.ones(shape=(data.shape[0], 1)), data, axis=1)

    # calculating the output and getting the label
    theta = np.exp(np.matmul(data, W))
    theta = theta / (np.sum(theta, axis=1).reshape((data.shape[0], 1)))
    label = theta.argmax(axis=1).reshape(data.shape[0], 1)
    return label


if __name__ == "__main__":
    """
    Script for Logistic Regression
    """
    train_data, train_label, validation_data, validation_label, test_data, test_label = preprocess()

    # number of classes
    n_class = 10

    # number of training samples
    n_train = train_data.shape[0]

    # number of features
    n_feature = train_data.shape[1]

    Y = np.zeros((n_train, n_class))
    for i in range(n_class):
        Y[:, i] = (train_label == i).astype(int).ravel()

    # Logistic Regression with Gradient Descent
    W = np.zeros((n_feature + 1, n_class))
    initialWeights = np.zeros((n_feature + 1, 1))
    opts = {'maxiter': 100}
    for i in range(n_class):
        labeli = Y[:, i].reshape(n_train, 1)
        args = (train_data, labeli)
        nn_params = minimize(blrObjFunction, initialWeights, jac=True, args=args, method='CG', options=opts)
        W[:, i] = nn_params.x.reshape((n_feature + 1,))

    # Find the accuracy on Training Dataset
    predicted_label = blrPredict(W, train_data)
    print('\n Training set Accuracy:' + str(100 * np.mean((predicted_label == train_label).astype(float))) + '%')

    # Find the accuracy on Validation Dataset
    predicted_label = blrPredict(W, validation_data)
    print('\n Validation set Accuracy:' + str(100 * np.mean((predicted_label == validation_label).astype(float))) + '%')

    # Find the accuracy on Testing Dataset
    predicted_label = blrPredict(W, test_data)
    print('\n Testing set Accuracy:' + str(100 * np.mean((predicted_label == test_label).astype(float))) + '%')

    with open ('params.pickle','wb') as f_parameter:
        pickle.dump([W],f_parameter)
    """
    Script for Support Vector Machine
    """

    print('\n\n--------------SVM-------------------\n\n')
    # number of training samples
    n_train_valid = validation_data.shape[0]
    n_train_test = test_data.shape[0]

    train_label_reshaped = train_label.reshape(n_train, )
    validation_label_reshaped = validation_label.reshape(n_train_valid, )
    test_label_reshaped = test_label.reshape(n_train_test, )

    #### LINEAR SVM
    linear_svc = SVC(kernel='linear')

    linear_svc.fit(train_data, train_label_reshaped)

    print('\nLinear SVM Accuracy:')

    print('Training set Accuracy:' + str(100 * linear_svc.score(train_data, train_label_reshaped)) + '%')
    print('Validation set Accuracy:' + str(100 * linear_svc.score(validation_data, validation_label_reshaped)) + '%')
    print('Testing set Accuracy:' + str(100 * linear_svc.score(test_data, test_label_reshaped)) + '%')

    ##### RADIAL SVM GAMMA 0.1
    radial_svm_1 = SVC(kernel='rbf', gamma=0.1)

    radial_svm_1.fit(train_data, train_label_reshaped)

    print('\nRADIAL SVM GAMMA 0.1 Accuracy:')

    print('Training set Accuracy:' + str(100 * radial_svm_1.score(train_data, train_label_reshaped)) + '%')
    print('Validation set Accuracy:' + str(100 * radial_svm_1.score(validation_data, validation_label_reshaped)) + '%')
    print('Testing set Accuracy:' + str(100 * radial_svm_1.score(test_data, test_label_reshaped)) + '%')

    #### RADIAL SVM GAMMA DEFAULT
    radial_svm_2 = SVC(kernel='rbf')

    radial_svm_2.fit(train_data, train_label_reshaped)

    print('\nRADIAL SVM Accuracy:')

    print('Training set Accuracy:' + str(100 * radial_svm_2.score(train_data, train_label_reshaped)) + '%')
    print('Validation set Accuracy:' + str(100 * radial_svm_2.score(validation_data, validation_label_reshaped)) + '%')
    print('Testing set Accuracy:' + str(100 * radial_svm_2.score(test_data, test_label_reshaped)) + '%')

    #### RADIAL SVM VARYING C VALUES
    # NOTE: default value of C is 1, therefore we can use the RADIAL SVM GAMMA DEFAULT as the first value

    print('\nRADIAL SVM Varying C value Accuracy:')

    C_ = 10.0
    for i in range(10):
        print('C = ' + str(C_))

        radial_svm_3 = SVC(C=C_, kernel='rbf')

        radial_svm_3.fit(train_data, train_label_reshaped)

        print('Training set Accuracy:' + str(100 * radial_svm_3.score(train_data, train_label_reshaped)) + '%')
        print('Validation set Accuracy:' + str(
            100 * radial_svm_3.score(validation_data, validation_label_reshaped)) + '%')
        print('Testing set Accuracy:' + str(100 * radial_svm_3.score(test_data, test_label_reshaped)) + '%')

        C_ = C_ + 10

    print('\n\n--------------SVM END-------------------\n\n')


    """
    Script for Extra Credit Part
    """
    # FOR EXTRA CREDIT ONLY
    print('\n\n--------------Multilabel Logistic Regression-------------------\n\n')
    W_b = np.zeros((n_feature + 1, n_class))
    initialWeights_b = np.zeros((n_feature + 1, n_class))
    opts_b = {'maxiter': 100}

    args_b = (train_data, Y)
    nn_params = minimize(mlrObjFunction, initialWeights_b, jac=True, args=args_b, method='CG', options=opts_b)
    W_b = nn_params.x.reshape((n_feature + 1, n_class))

    # Find the accuracy on Training Dataset
    predicted_label_b = mlrPredict(W_b, train_data)
    print('\n Training set Accuracy:' + str(100 * np.mean((predicted_label_b == train_label).astype(float))) + '%')

    # Find the accuracy on Validation Dataset
    predicted_label_b = mlrPredict(W_b, validation_data)
    print('\n Validation set Accuracy:' + str(100 * np.mean((predicted_label_b == validation_label).astype(float))) + '%')

    # Find the accuracy on Testing Dataset
    predicted_label_b = mlrPredict(W_b, test_data)
    print('\n Testing set Accuracy:' + str(100 * np.mean((predicted_label_b == test_label).astype(float))) + '%')
