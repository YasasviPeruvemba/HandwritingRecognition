import pickle
import os
import numpy as np
import pandas as pd
import urllib.request
import scipy.sparse
import matplotlib.pyplot as plt

def load_dataset():
    def download(filename,source='http://yann.lecun.com/exdb/mnist/'):
        # function to download data from mnist website
        print ('Downloading ',filename)
        urllib.request.urlretrieve(source+filename,filename)

    import gzip

    def load_images(filename):
        if not os.path.exists(filename):
            download(filename)
        # if data does not exist on the computer, it downloads it
        # we proceed to zip it now

        with gzip.open(filename,'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=16)
            # the above line makes a 1d integer numpy array of all the images in the dataset
            # we now reshape the data to have 60,000 images of size 28x28 each
            #x = len(data)/784;
            data = np.reshape(data,(-1,784))
            #print(x)
            # -1 is used for the np array to infer the number of images from the dataset itself
            # 1 is the number of channels, i.e colour types
            # 28 and 28 are the dimansions of the image
        return data/np.float32(256)
        # this ensures data returned is in float form, also it returns a 4d array

    def load_labels(filename):
        if not os.path.exists(filename):
            download(filename)
        # if data does not exist on the computer, it downloads it
        # we proceed to zip it now
        
        with gzip.open(filename,'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=8)

        return data

    X_Train = load_images('train-images-idx3-ubyte.gz')
    Y_Train = load_labels('train-labels-idx1-ubyte.gz')
    X_Test = load_images('t10k-images-idx3-ubyte.gz')
    Y_Test = load_labels('t10k-labels-idx1-ubyte.gz')

    return X_Train, Y_Train, X_Test, Y_Test

#print('Here')
X_Train, Y_Train, X_Test, Y_Test = load_dataset()
#print('Now here')
# Seeing how the test images look like
#plt.imshow(X_Train[0][0])
#plt.show()

def Loss(w,x,y,lam):
    m = x.shape[0]
    y_mat = oneHotIt(y)
    scores = np.dot(x,w)
    prob = softmax(scores)
    loss = (-1 / m) * np.sum(y_mat * np.log(prob)) + (lam/2)*np.sum(w*w)
    grad = (-1 / m) * np.dot(x.T,(y_mat - prob)) + lam*w
    return loss, grad

def oneHotIt(Y):
    m = Y.shape[0]
    OHX = scipy.sparse.csr_matrix((np.ones(m),(Y, np.array(range(m)))))
    OHX = np.array(OHX.todense()).T
    return OHX

def softmax(x):
    x = x - np.max(x)
    sm = (np.exp(x).T/ np.sum(np.exp(x),axis=1)).T
    return sm

def ProbsAndPreds(X):
    probs = softmax(np.dot(X,w))
    preds = np.argmax(probs,axis=1)
    return probs, preds

print('X_Train.shape[0] : ',X_Train.shape[0])
print('X_Train.shape[1] : ',X_Train.shape[1])

# Use the below commented line if you have not created the pickle file for the first time
#w = np.zeros([784,len(np.unique(Y_Train))])

# Loading saved values
pickle_in = open('HandwritingRecogWeights.pickle','rb')
w = pickle.load(pickle_in)

lam = 1
iterations = 1000
learningRate = 1e-3
losses = []

for i in range(0,iterations):
    loss, grad = Loss(w,X_Train,Y_Train,lam)
    losses.append(loss)
    w = w - (learningRate * grad)
    print('Loss for',i,'th iteration is',loss)

print('Final Loss is :',loss)
plt.plot(losses)
plt.show()

def Accuracy(X,Y):
    prob,prede = ProbsAndPreds(X)
    accuracy = sum(prede == Y)/float(len(Y))
    return accuracy

print('Training Accuracy :',Accuracy(X_Train,Y_Train))
print('Testing Accuracy :',Accuracy(X_Test,Y_Test))

#Save the weights for future use
pickle_out = open('HandwritingRecogWeights.pickle','wb')
pickle.dump(w,pickle_out)
pickle_out.close()

plt.imshow(scipy.reshape(w[:,0],[28,28]))
plt.show()
plt.imshow(scipy.reshape(w[:,1],[28,28]))
plt.show()
plt.imshow(scipy.reshape(w[:,2],[28,28]))
plt.show()
plt.imshow(scipy.reshape(w[:,3],[28,28]))
plt.show()
plt.imshow(scipy.reshape(w[:,4],[28,28]))
plt.show()
plt.imshow(scipy.reshape(w[:,5],[28,28]))
plt.show()
plt.imshow(scipy.reshape(w[:,6],[28,28]))
plt.show()
plt.imshow(scipy.reshape(w[:,7],[28,28]))
plt.show()
plt.imshow(scipy.reshape(w[:,8],[28,28]))
plt.show()
plt.imshow(scipy.reshape(w[:,9],[28,28]))
plt.show()
