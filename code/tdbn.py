"""
Theano Temporal Deep Belief Network implementation (TDBN) used for temporal pattern recognition
This model based on Sukhbaatar paper :
Robust generation of dynamical patterns in human motion by a deep belief nets
[http://jmlr.csail.mit.edu/proceedings/papers/v20/sukhbaatar11/sukhbaatar11.pdf]

Sukhbaatar uses this model for gesture generation. In our case, TDBNs used for gesture recognition.
To do that, we stacked a regressive layer on top of a TDBN, a new layer who is in charge of supervised.

@author Francois Lasson : CERV Brest France
@mail : francois.lasson@gmail.com
"""

#Numpy and Theano library
import numpy as np
#Import pyPlot to display cost evolution during learning phase
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
import theano
import theano.tensor as T
#Use to compute learning time
import timeit
#Use to save trained model or dataset
import cPickle
#This class used to generate dataset readable by RBMs from BVH files
from motion import generate_dataset
#Import RBM Class, lower level of TDBN
from rbm import RBM, create_train_rbm
#Import CRBMLogistic Class, upper level of TDBN (CRBM+Logistic regressive layer)
from CRBMLogistic import LOGISTIC_CRBM, create_train_LogisticCrbm


class TDBN(object):
    """ Structure of a Temporal Deep Belief Network used for discrimination (TDBN)
    For rbms, indexes are the following : 0 = trunk; 1 = left arm; 2 = right Arm; 3 = left leg; 4 = right leg"""
    def __init__(self, rbms=None, log_crbm = None):
        #A TDBN is composed of five RBMs which are used to extract spatial features (Cf RBM class)
        self.rbms = [rbms[0],rbms[1],rbms[2],rbms[3],rbms[4]] #TODO Errror if self.rbms = [] then loop for due to length of list, tdbn.pkg become too huge
        #At the top, there is a CRBM (Conditional RBM) combined with a Logistic regressive. The CRBM is used for temporal features extraction. For its part, the Logistic regressive
        #allow us to work with labeled datas, in other word, to do a supervised training. (Cf CRBMLogistic Class)
        self.log_crbm = log_crbm

    """The recognize_file function is used to predict labels of each frames of a file using a minibatch"""
    def recognize_file_with_minibatch(self, files=None, real_labels = None):
        #First step : get dataset from file
        data_rbms_0, data_rbms_1, data_rbms_2, data_rbms_3, data_rbms_4, labelset, seqlen = generate_dataset(files, real_labels)
        data_rbms = [data_rbms_0, data_rbms_1, data_rbms_2, data_rbms_3, data_rbms_4]
        #Now, get the number of frames in this dataset (in the file we have to recognize)
        number_of_frames = data_rbms[0].get_value(borrow=True).shape[0]
        #For each frames, recognize label
        dataset_crbm = [] #generate a new dataset test for CRBM from RBMs
        for index_frame in range(number_of_frames):
            #propup frame on each RBMs
            hidden_rbms = []
            for index_rbm in range(5):
                frame = data_rbms[index_rbm].get_value(borrow=True)[index_frame]
                hidden_rbms.append(self.rbms[index_rbm].predict_hidden(frame))
            #concatenate hidden RBMs'layer
            concatenate_hidden_rbms = np.concatenate(hidden_rbms, axis=0)
            #append to dataset
            dataset_crbm.append(concatenate_hidden_rbms)
            #print information about % of generation for user
            print "CRBM dataset test generation : %d / %d " %(index_frame,number_of_frames)
        shared_dataset_crbm = theano.shared(np.asarray(dataset_crbm, dtype=theano.config.floatX))
        #Send dataset to logCRBM to get recognized label
        self.log_crbm.recognize_dataset(dataset_test=shared_dataset_crbm, seqlen = seqlen)

    """The recognize_file function is used to predict labels of each frames of a file, frame by frame"""
    def recognize_files_frame_by_frame(self, file=None, real_label = None):
        #confusion matrix : https://ccrma.stanford.edu/workshops/mir2009/references/ROCintro.pdf
        confusion_matrix = np.zeros((self.log_crbm.n_label,self.log_crbm.n_label)) #REAL : column, #Predict : line
        #get time for recognition
        start_time = timeit.default_timer()
        #First step : get dataset from file
        data_rbms_0, data_rbms_1, data_rbms_2, data_rbms_3, data_rbms_4, labelset, seqlen = generate_dataset(file, real_label)
        data_rbms = [data_rbms_0, data_rbms_1, data_rbms_2, data_rbms_3, data_rbms_4]
        #Now, get the number of frames in this dataset (in the file we have to recognize)
        number_of_frames = data_rbms[0].get_value(borrow=True).shape[0]

        #Initialize past visible layers (memory)
        past_visible = []
        for i in range(self.log_crbm.delay):
            past_visible.append(np.zeros(self.log_crbm.n_input))
        concatenate_past_visible = np.concatenate(past_visible, axis=0)

        if (number_of_frames < self.log_crbm.delay*self.log_crbm.freq):
            return
        else :
            print "The %d first frames used to initialize past visibles layers (ignored for PER)" %(self.log_crbm.delay*self.log_crbm.freq)
        for index_frame in range(self.log_crbm.delay*self.log_crbm.freq,number_of_frames):
            #propup this frame on each RBMs to get hidden representations
            hidden_rbms = []
            for index_rbm in range(5):
                hidden_rbms.append(self.rbms[index_rbm].predict_hidden(data_rbms[index_rbm].get_value(borrow=True)[index_frame]))
            concatenate_hidden_rbms = np.concatenate(hidden_rbms, axis=0)

            #prop up to CRBM logistic
            predict_label = self.log_crbm.predict_label(concatenate_hidden_rbms, concatenate_past_visible)
            print "Frame %d : Real label %d; label found %d" %(index_frame, labelset[index_frame], predict_label)
            #actualize confusion matrix
            confusion_matrix[predict_label,labelset[index_frame]]+=1
            #Actualize past visible layers (memory)
            for i in range(self.log_crbm.delay-1, 0, -1):
                past_visible[i] = past_visible[i-1]
            past_visible[0] = concatenate_hidden_rbms
            concatenate_past_visible = np.concatenate(past_visible, axis=0)

        end_time = timeit.default_timer()
        recognition_time = (end_time - start_time)
        PER=0
        print ('Time for recognition : %f secondes ; Duration of file : %f secondes.' %(recognition_time,number_of_frames/120.))
        return confusion_matrix


    """The recognize_file function is used to predict labels of chunks and determine PER(several frames)"""
    def recognize_files_chunk(self, file=None, real_label = None):
        #recognize frames using chunk and average value on chunk
        chunk_size = 240 #number of frames in a chunk
        chunk_covering = 30 #number of frames for covering

        #confusion matrix : https://ccrma.stanford.edu/workshops/mir2009/references/ROCintro.pdf
        confusion_matrix = np.zeros((self.log_crbm.n_label,self.log_crbm.n_label)) #REAL : column, #Predict : line
        #get time for recognition
        start_time = timeit.default_timer()
        #First step : get dataset from file
        data_rbms_0, data_rbms_1, data_rbms_2, data_rbms_3, data_rbms_4, labelset, seqlen = generate_dataset([file], [real_label])
        data_rbms = [data_rbms_0, data_rbms_1, data_rbms_2, data_rbms_3, data_rbms_4]
        #Now, get the number of frames in this dataset (in the file we have to recognize)
        number_of_frames = data_rbms[0].get_value(borrow=True).shape[0]

        #Initialize past visible layers (memory) at Null
        past_visible = []
        for i in range(self.log_crbm.delay):
            past_visible.append(np.zeros(self.log_crbm.n_input))
        concatenate_past_visible = np.concatenate(past_visible, axis=0)

        #For each frames, predict label
        chunk_predict_tab = []
        chunk_real_tab = []
        if (number_of_frames < self.log_crbm.delay*self.log_crbm.freq):
            return
        else :
            print "The %d first frames used to initialize past visibles layers (ignored for PER)" %(self.log_crbm.delay*self.log_crbm.freq)
        for index_frame in range(self.log_crbm.delay*self.log_crbm.freq,number_of_frames):
            #propup this frame on each RBMs to get hidden representations
            hidden_rbms = []
            for index_rbm in range(5):
                hidden_rbms.append(self.rbms[index_rbm].predict_hidden(data_rbms[index_rbm].get_value(borrow=True)[index_frame]))
            concatenate_hidden_rbms = np.concatenate(hidden_rbms, axis=0)
            #prop up to CRBM logistic
            predict_label = self.log_crbm.predict_label(concatenate_hidden_rbms, concatenate_past_visible)
            #print predict for this frame
            print "Frame %d : Real label %d; label found %d" %(index_frame, labelset[index_frame], predict_label)
            #add to chunk tab
            chunk_predict_tab.append(int(predict_label))
            chunk_real_tab.append(labelset[index_frame])
            #Actualize past visible layers (memory)
            for i in range(self.log_crbm.delay-1, 0, -1):
                past_visible[i] = past_visible[i-1]
            past_visible[0] = concatenate_hidden_rbms
            concatenate_past_visible = np.concatenate(past_visible, axis=0)

            if(index_frame%chunk_size==0 and index_frame!=self.log_crbm.delay*self.log_crbm.freq):
                predict_chunk = 0
                real_chunk = 0
                #get the average predict label and real label on this chunk (chunk_size frames)
                for label in range(1,self.log_crbm.n_label):
                        if(chunk_predict_tab.count(label)>chunk_predict_tab.count(predict_chunk)):
                            predict_chunk=label
                        if(chunk_real_tab.count(label)>chunk_real_tab.count(real_chunk)):
                            real_chunk=label
                print "-> Chunk : Real label %d; label found %d \n" %(real_chunk, predict_chunk)
                #actualize confusion matrix
                confusion_matrix[predict_chunk,real_chunk]+=1
                #clean chunk tabs
                chunk_predict_tab[:]=[]
                chunk_real_tab[:]=[]
        end_time = timeit.default_timer()
        recognition_time = (end_time - start_time)
        print ('Time for recognition : %f secondes ; Duration of file : %f secondes.' %(recognition_time,number_of_frames/120.))
        return confusion_matrix


def create_train_tdbn(training_files=None, training_labels = None,
                      validation_files=None, validation_labels = None,
                      test_files=None, test_labels = None,
                      rbm_training_epoch = 10000, rbm_learning_rate=1e-3, rbm_n_hidden=30, batch_size = 100,
                      crbm_training_epoch = 10000, crbm_learning_rate = 1e-3, crbm_n_hidden = 300, crbm_n_delay=6, crbm_freq=10,
                      finetune_epoch = 10000, finetune_learning_rate = 0.1, log_n_label=9):

    #Train or load? User have choice in case where *.pkl (pretrain models saves) exist
    """Do you want to retrain all rbms? crbm? regenerate crbm dataset? This step must be done in case of a new dataset"""
    retrain_rbms = True
    regenerate_crbm_dataset = True
    retrain_crbm = True

    #First step : generate dataset from files
    data_rbms_0, data_rbms_1, data_rbms_2, data_rbms_3, data_rbms_4, labelset, seqlen = generate_dataset(training_files, training_labels)
    data_rbms = [data_rbms_0, data_rbms_1, data_rbms_2, data_rbms_3, data_rbms_4]

    #In order to have an idea of cost evolution during training phase (convergence, oscilation, etc.) we save their prints in a plot.
    #So, we have create and initialize a pyplot
    if (retrain_rbms) :
        title = "RBMs training phase : \n epoch="+str(rbm_training_epoch)+"; Number of hidden= "+str(rbm_n_hidden)+"; Learning rate="+str(rbm_learning_rate)
        plt.title(title)
        plt.ylabel("Cost")
        plt.xlabel("Epoch")
        x= xrange(rbm_training_epoch)

    #Now, if user have choosen to retrain RBMs, we train new RBMs from the datasets previously generated. Else, we just reload RBMs from pkl files
    rbms = []
    for index_rbm in range(5):
        rbm_filename = "trained_model/Pretrain_RBM_" + str(index_rbm) + ".pkl"
        if (retrain_rbms) :
            rbm, cost_rbms = create_train_rbm(dataset = data_rbms[index_rbm], seqlen = seqlen, training_epochs=rbm_training_epoch,
                                             learning_rate = rbm_learning_rate, batch_size=batch_size, n_hidden=rbm_n_hidden)
            rbms.append(rbm)
            plt.plot(x,cost_rbms, label="RBM"+str(index_rbm))
            with open(rbm_filename, 'w') as f:
                cPickle.dump(rbm, f)
        else :
            rbm = cPickle.load(open(rbm_filename))
            rbms.append(rbm)
    #save plot (name depend of time)
    if (retrain_rbms) :
        plt.legend(loc=4)
        plot_name = 'plot/rbm_'+str(timeit.default_timer())+'.png'
        plt.savefig(plot_name)
        print "RBMs plot is saved!"
        plt.clf() #without this line, all is plot in the same figure
        plt.close()
    #The next step is to generate a new dataset in order to train the CRBM. To do that, we realize a propagation-up of previous dataset in RBMs
    dataname = ['trained_model/dataset_train.pkl','trained_model/dataset_valid.pkl','trained_model/dataset_test.pkl']
    labelname = ['trained_model/labelset_train.pkl','trained_model/labelset_valid.pkl','trained_model/labelset_test.pkl']
    if(regenerate_crbm_dataset):
        #In fact, we have to generate three dataset : one for training phase, one for validation and one for test
        for i in range(3):
            if(i==0): #training phase
                files = training_files
                labels = training_labels
            elif(i==1): #validation
                files = validation_files
                labels = validation_labels
            else : #and test
                files = test_files
                labels = test_labels
            #generate dataset from file for each of these phases
            data_rbms_0, data_rbms_1, data_rbms_2, data_rbms_3, data_rbms_4, labelset, seqlen = generate_dataset(files, labels)
            data_rbms = [data_rbms_0, data_rbms_1, data_rbms_2, data_rbms_3, data_rbms_4]
            #propup dataset from visible to hidden layers of RBMs in order to generate data set for crbm
            dataset_crbm = []
            number_of_samples = data_rbms[0].get_value(borrow=True).shape[0] #all datas have same size, so data_trunk have been choosen but it could be replace by another
            for index_data in range(number_of_samples):
                hidden_rbms = []
                for index_rbm in range(5):
                    #propup the index_data frame on rbms
                    frame = data_rbms[index_rbm].get_value(borrow=True)[index_data]
                    prop_up = rbms[index_rbm].predict_hidden(frame)
                    hidden_rbms.append(prop_up)
                #concanate output of rbms in order to get only one frame for CRBM
                concatenated_hidden_rbms = np.concatenate((hidden_rbms[0], hidden_rbms[1], hidden_rbms[2], hidden_rbms[3], hidden_rbms[4]), axis=0)
                #add this to dataset_crbm
                dataset_crbm.append(concatenated_hidden_rbms)
                if(index_data%(number_of_samples//100)==0):
                    print ('CRBM dataset generation %d : %d %%' %(i,((100*index_data/number_of_samples)+1)) )

            with open(dataname[i], 'w') as f:
                cPickle.dump(dataset_crbm, f)
            with open(labelname[i], 'w') as f:
                cPickle.dump(labelset, f)

    #So, CRBM dataset is generated. Now we have to train Logistic_CRBM in order to create a TDBN
    start_time = timeit.default_timer() #Get time, this used to compute training time
    #get length of dataset. A dataset is generally generated from several files, the following variables inform of this point (For futher information Cf motion class)
    trainingLen = generate_dataset(training_files, training_labels)[6]
    validationLen = generate_dataset(validation_files, validation_labels)[6]
    testLen = generate_dataset(test_files, test_labels)[6]

    #load and generate shared_data and shared_label, theano optimization that we use to generate optimized dataset (memory optimization)
    shared_dataset_crbm = []
    shared_labelset_crbm = []
    for i in range(3):
        shared_dataset_crbm.append(theano.shared(np.asarray(cPickle.load(open(dataname[i])), dtype=theano.config.floatX)))
        shared_labelset_crbm.append(theano.shared(np.asarray(cPickle.load(open(labelname[i])))))

    #CRBMs
    title = "Pre-training : epoch="+str(crbm_training_epoch)+"; delay="+str(crbm_n_delay)+\
            "; n_hidden= "+str(crbm_n_hidden)+"; learning rate="+str(crbm_learning_rate)

    #At this step, we have enough elements to create a logistic regressive CRBM
    log_crbm, cost_crbm, PER_x_valid,PER_y_valid,PER_x_test,PER_y_test = create_train_LogisticCrbm(
                                        dataset_train=shared_dataset_crbm[0], labelset_train=shared_labelset_crbm[0], seqlen_train = trainingLen,
                                        dataset_validation=shared_dataset_crbm[1], labelset_validation=shared_labelset_crbm[1], seqlen_validation = validationLen,
                                        dataset_test=shared_dataset_crbm[2], labelset_test=shared_labelset_crbm[2], seqlen_test = testLen,
                                        batch_size = batch_size, pretraining_epochs=crbm_training_epoch, pretrain_lr = crbm_learning_rate,
                                        number_hidden_crbm = crbm_n_hidden, n_delay=crbm_n_delay, freq = crbm_freq,
                                        training_epochs = finetune_epoch, finetune_lr = finetune_learning_rate, n_label = log_n_label, retrain_crbm = retrain_crbm)
    #Plot CRBM influence PER evolution
    font = {'size':'11'}
    title += "\nFine-tuning : epoch="+str(finetune_epoch)+"; Learning rate="+str(finetune_learning_rate)
    plt.subplot(211)
    plt.title(title, **font)
    plt.ylabel("PER_validation")
    plt.plot(PER_x_valid,PER_y_valid)
    plt.subplot(212)
    plt.ylabel("PER_test")
    plt.xlabel("epoch")
    plt.plot(PER_x_test,PER_y_test)
    plot_name = 'plot/PER_crbm_'+str(timeit.default_timer())+'.png'
    plt.savefig(plot_name)
    print "CRBM PER plot is saved!"
    plt.clf() #without this line, all is plot in the same figure
    plt.close()
    #Get training time to inform user
    end_time = timeit.default_timer()
    pretraining_time = (end_time - start_time)
    print ('Temporal DBN : Training took %f minutes' % (pretraining_time / 60.))
    #Last step, create and return Temporal Deep Belief Net
    tdbn = TDBN(rbms, log_crbm)
    return tdbn

def confusion_matrix_analysis(confusion_matrix=None, n_labels=None):
    print "\nConfusion matrix :"
    print confusion_matrix
    number_of_frames = np.sum(confusion_matrix)
    print "Number of frames in the test dataset : %d" %number_of_frames
    print "\n->Analysis :"
    #for each gestures (label)
    for i in range(n_labels):
        print "Gesture label %d (gesture%d*.bvh):" %(i,i+1)
        #determine confusion matrix
        confusion_matrix_gesture = np.zeros((2,2))
        TP,FN,FP,TN=0,0,0,0
        TP = confusion_matrix[i,i]
        FP = np.sum(confusion_matrix[:,i])-TP
        FN = np.sum(confusion_matrix[i,:])-TP
        TN = number_of_frames-(TP+FP+FN)

        confusion_matrix_gesture[0,0] = TP
        confusion_matrix_gesture[1,0] = FP
        confusion_matrix_gesture[0,1] = FN
        confusion_matrix_gesture[1,1] = TN
        print confusion_matrix_gesture
        precision = TP/(TP+FP)
        recall = TP/(TP+FN)
        accuracy = (TP+TN)/number_of_frames
        f_measure = 2/((1/precision)+(1/recall))
        print "Nb frames : %d" %(TP+FP)
        print "Precision : %f %%" %(precision*100)
        print "Recall : %f %%" %(recall*100)
        print "Accuracy : %f %%" %(accuracy*100)
        print "F_measure : %f %%" %(f_measure*100)
        print "-------------"
    PER = 1 - np.sum(np.diag(confusion_matrix))/number_of_frames
    print "PER = %f %%" %(PER*100)

if __name__ == '__main__':
    #Firstly, do you want to train a TDBN or do you want to test it
    do_training_phase = True
    #To create a TDBN, we have to train our model, validate his training and test it
    #To do that, we have to define BVH files use to create datasets and their associate labels
    training_files = ['data/geste1a.bvh','data/geste1b.bvh','data/geste1c.bvh', 'data/geste1d.bvh',
                      'data/geste2a.bvh','data/geste2b.bvh','data/geste2c.bvh', 'data/geste2d.bvh',
                      'data/geste3a.bvh','data/geste3b.bvh','data/geste3c.bvh', 'data/geste3d.bvh',
                      'data/geste13a.bvh','data/geste13b.bvh','data/geste13c.bvh', 'data/geste13d.bvh',
                      'data/geste5a.bvh','data/geste5b.bvh','data/geste5c.bvh', 'data/geste5d.bvh',
                      'data/geste14a.bvh','data/geste14b.bvh','data/geste14c.bvh', 'data/geste14d.bvh',
                      'data/geste18a.bvh','data/geste18b.bvh','data/geste18c.bvh', 'data/geste18d.bvh',
                      'data/geste8a.bvh','data/geste8b.bvh','data/geste8c.bvh', 'data/geste8d.bvh',
                      'data/geste10a.bvh','data/geste10b.bvh','data/geste10c.bvh', 'data/geste10d.bvh']
    training_labels = [0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3,4,4,4,4,5,5,5,5,6,6,6,6,7,7,7,7,8,8,8,8]

    validation_files = ['data/geste1e.bvh','data/geste1f.bvh','data/geste1g.bvh','data/geste1h.bvh',
                        'data/geste2e.bvh','data/geste2f.bvh','data/geste2g.bvh','data/geste2h.bvh',
                        'data/geste3e.bvh','data/geste3f.bvh','data/geste3g.bvh','data/geste3h.bvh',
                        'data/geste13e.bvh','data/geste13f.bvh','data/geste13g.bvh','data/geste13h.bvh',
                        'data/geste5e.bvh','data/geste5f.bvh','data/geste5g.bvh','data/geste5h.bvh',
                        'data/geste14e.bvh','data/geste14f.bvh','data/geste14g.bvh','data/geste14h.bvh',
                        'data/geste18e.bvh','data/geste18f.bvh','data/geste18g.bvh','data/geste18h.bvh',
                        'data/geste8e.bvh','data/geste8f.bvh','data/geste8g.bvh','data/geste8h.bvh',
                        'data/geste10a.bvh','data/geste10a.bvh','data/geste10g.bvh','data/geste10h.bvh']
    validation_labels = [0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3,4,4,4,4,5,5,5,5,6,6,6,6,7,7,7,7,8,8,8,8]

    test_files = ['data/geste1i.bvh',
                  'data/geste2i.bvh',
                  'data/geste3i.bvh',
                  'data/geste13i.bvh',
                  'data/geste5i.bvh',
                  'data/geste14i.bvh',
                  'data/geste18i.bvh',
                  'data/geste8i.bvh',
                  'data/geste10i.bvh']
    test_labels = [0,1,2,3,4,5,6,7,8]

    #Here, there are two possibilities :
    #You're prevously chosen to train a TDBN, so create it and save it
    if do_training_phase:
        #create and train a tdbn
        tdbn = create_train_tdbn(
                            training_files = training_files, training_labels = training_labels,
                            validation_files= validation_files, validation_labels = validation_labels,
                            test_files = test_files, test_labels = test_labels)
        #save our network
        with open('trained_model/best_model_tdbn.pkl', 'w') as f:
            cPickle.dump(tdbn, f)

    #Else, you have already trained a TDBN and you want to test it
    else :
        print "Load TDBN..."
        tdbn = cPickle.load(open('trained_model/best_model_tdbn.pkl'))
        print "TDBN loaded from file."
        #tdbn.recognize_file_with_minibatch(test_files, test_labels) #using minibatch

        confusion_matrix = np.zeros((tdbn.log_crbm.n_label,tdbn.log_crbm.n_label)) #REAL : column, #Predict : line
        for file_index in range(len(test_files)):
            print "Filename : "+test_files[file_index]+ " (label %d)" %test_labels[file_index]
            confusion_matrix += tdbn.recognize_files_chunk(test_files[file_index], test_labels[file_index])
            #confusion_matrix += tdbn.recognize_files_frame_by_frame(test_files[file_index], test_labels[file_index])
        confusion_matrix_analysis(confusion_matrix, tdbn.log_crbm.n_label)
