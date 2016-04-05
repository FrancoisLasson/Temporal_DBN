"""
This python file provide some methods to load motion capture data (Mocap).
In our case, mocap are encoded as BVH files but there functions are
could be easily adapt to others formats.

@author Francois Lasson : CERV Brest France
@mail : francois.lasson@gmail.com
"""

import numpy as np
import theano

#Read external file (BVH file)
import scipy.io

#This function used to normalize value contains in a BVH file in order to make it
#readable by a gaussian RBM net. In fact, a Gaussian RBM is a RBM ables to work with continuous data.
#Yet, data have to be in [0,1] however, our bvh values are angles, so i.e. in range [0,360]
def normalize_value(value) :
    value = value%360 #centre on [0,360]
    value = value/360 #reduce to [0,1]
    if (value>1 or value<0):
        print "error with data normalization (cf normalize_value func)"
        return None
    return value

#Main Function of file, it used to read BVH files and create dataset
def generate_dataset(filenames, labels):
    #if filenames & labels don't have same length : error
    if len(filenames) != len(labels) :
        print "Error in dataset generation : please verify that each file have one and only one label"

     #tabs of tabs which contains all frames of all files
    datasetTrunk = []
    datasetLeftArm = []
    datasetRightArm = []
    datasetLeftLeg = []
    datasetRigthLeg = []
    seqlen = [] #size of sequences
    labelset = [] #labels for each frames

    for index in range(len(filenames)):
        #open file
        f = open(filenames[index],'r')
        #ignore first part : skeleton definiton
        cond = False
        while cond is False:
            line = f.readline()
            if line[:6] =='MOTION':
                cond = True

        line = f.readline()#Ignore next line
        line = f.readline()#Ignore next line
        #Get motion data in the next part of file
        numberOfFrames = 0
        line = f.readline() #read line
        while line: #for each line
            values = map(float, line.split()) #divide line in several values
            neurones = np.array([]) #Create np.array for neurones
            for index_value in range(len(values)):
                value = normalize_value(values[index_value]) #get value and normalize it #values[index_value]
                neurones = np.append(neurones, value) #add value to array

            #This part depends of our skeleton hierarchy : read first part of the bvh file
            """CMU DATA BASE """
            """
            indexTrunkBones = np.r_[np.arange(3,6),np.arange(36,54)]
            datasetTrunk.append(neurones[indexTrunkBones])#21 neurones
            datasetLeftArm.append(neurones[6:21])#15 neurones
            datasetRightArm.append(neurones[21:36]) #15 neurones
            datasetLeftLeg.append(neurones[54:75]) #21 neurones
            datasetRigthLeg.append(neurones[75:96]) #21 neurones
            """
            """CERV MOCAP DATABASE """

            datasetTrunk.append(neurones[3:18])#15 neurones
            datasetLeftArm.append(neurones[18:30])#12 neurones
            datasetRightArm.append(neurones[30:42]) #12 neurones
            datasetLeftLeg.append(neurones[42:54]) #12 neurones
            datasetRigthLeg.append(neurones[54:66]) #12 neurones

            #label
            labelset.append(labels[index])
            numberOfFrames+=1
            line = f.readline()
        seqlen.append(numberOfFrames)
    # put data into shared memory
    shared_trunk = theano.shared(np.asarray(datasetTrunk, dtype=theano.config.floatX))
    shared_leftArm = theano.shared(np.asarray(datasetLeftArm, dtype=theano.config.floatX))
    shared_rightArm = theano.shared(np.asarray(datasetRightArm, dtype=theano.config.floatX))
    shared_leftLeg = theano.shared(np.asarray(datasetLeftLeg, dtype=theano.config.floatX))
    shared_rightLeg = theano.shared(np.asarray(datasetRigthLeg, dtype=theano.config.floatX))
    print "Files dataset has been generated."
    return shared_trunk,shared_leftArm,shared_rightArm,shared_leftLeg,shared_rightLeg, labelset, seqlen


if __name__ == "__main__":
    print "Dataset generation class"
    files = ['data/geste1b.bvh','data/geste2b.bvh']
    labels = [1,0]
    trunk, larm, rarm,lleg, rleg, labelset, seqlen = generate_dataset(files, labels)
