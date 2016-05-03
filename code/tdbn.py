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
from theano import ProfileMode
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

profmode = theano.ProfileMode(optimizer='fast_run', linker=theano.gof.OpWiseCLinker())

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

    """
    The recognize_file function is used to predict labels of chunks and determine PER(several frames)
    Function with optimization for a GPU execution
    """
    def recognize_files_chunk(self, file=None, real_label = None):
        #recognize frames using chunk and average value on chunk
        chunk_size = 1 #number of frames in a chunk
        chunk_covering = 0  #number of frames for covering
        #to recognize frame by frame : chunk_size=1 and chunk_covering=0

        #confusion matrix : https://ccrma.stanford.edu/workshops/mir2009/references/ROCintro.pdf
        confusion_matrix = np.zeros((self.log_crbm.n_label,self.log_crbm.n_label), theano.config.floatX) #REAL : column, #Predict : line

        #compute functions used to propup data into our model : Optimized process
        shared_input_tab = [] #shared inputs for our 5 rbms
        for index_rbm in range(5):
            input_type = np.zeros(self.rbms[index_rbm].n_visible).astype(theano.config.floatX)
            shared_input_tab.append(theano.shared(input_type.astype(theano.config.floatX)))
        #shared variable for the crbm history
        history_type = np.zeros(self.log_crbm.delay*self.log_crbm.n_input).astype(theano.config.floatX)
        shared_history = theano.shared(history_type.astype(theano.config.floatX))

        #function to propup: process from RBMs to label
        #To optimize the execution time, prefer a use a shared varaible rather than theano symbolic variable
        output_rbm_sample = []
        #actualize value of the shared variables
        for index_rbm in range(5):
            output_sigmoid_value = T.nnet.sigmoid(T.dot(shared_input_tab[index_rbm], self.rbms[index_rbm].W) + self.rbms[index_rbm].hbias)
            output_rbm_sample.append(self.rbms[index_rbm].theano_rng.binomial(size=output_sigmoid_value.shape,n=1, p=output_sigmoid_value,dtype=theano.config.floatX))
        concatenate_hidden_rbms = T.concatenate(output_rbm_sample, axis=0)
        output_crbm = T.nnet.sigmoid(T.dot(concatenate_hidden_rbms, self.log_crbm.crbm_layer.W)+ T.dot(shared_history, self.log_crbm.crbm_layer.B) + self.log_crbm.crbm_layer.hbias)
        output_log = T.nnet.softmax(T.dot(output_crbm, self.log_crbm.logLayer.W) + self.log_crbm.logLayer.b)
        prediction = T.argmax(output_log, axis=1)[0]
        prop_up_func = theano.function([], [concatenate_hidden_rbms,prediction])

        #First step : get dataset from file
        data_rbms_0, data_rbms_1, data_rbms_2, data_rbms_3, data_rbms_4, labelset, seqlen = generate_dataset([file], [real_label])
        data_rbms = [data_rbms_0, data_rbms_1, data_rbms_2, data_rbms_3, data_rbms_4]
        #Now, get the number of frames in this dataset (in the file we have to recognize)
        number_of_frames = data_rbms[0].get_value(borrow=True).shape[0]
        #Check size : if this file do not have enough frames to Initialize past visibles, ignore this file
        nb_frames_initialization = self.log_crbm.delay*self.log_crbm.freq #number of frames required for past visibles initialization
        if (number_of_frames < nb_frames_initialization):
            return
        #Initialize past visible layers (memory) at Null
        concatenate_past_visible = np.zeros(self.log_crbm.n_input*self.log_crbm.delay, theano.config.floatX)
        #list of chunks : every chunks are present in these list
        all_chunks_predict = []
        all_chunks_real = []

        #get time for recognition : recognition start here
        start_time = timeit.default_timer()
        #for each frames
        for index_frame in range(number_of_frames):
            #create a new chunk each chunk_size-chunk_covering
            if((index_frame-nb_frames_initialization)%(chunk_size-chunk_covering)==0 and index_frame>=nb_frames_initialization):
                    all_chunks_predict.append([])
                    all_chunks_real.append([])
            #Actualize shared variable values
            for index_rbm in range(5):
                shared_input_tab[index_rbm].set_value(data_rbms[index_rbm].get_value(borrow=True)[index_frame])
            shared_history.set_value(concatenate_past_visible)
            #call our theano function (which one uses shared variables) to get label and output rbms we use to actualize past_visible (memory)
            concatenate_hidden_rbms, predict_label = prop_up_func()
            #Actualize past visible layers (memory)
            #split past visibles : in our implementation, past visible is a big vector (size = delay*crbm.n_input)
            #in order to actualize past visibles layers, a nice way is to split this big vector, Actualize each subvectors and concatenate these subvectors
            past_visible = np.split(concatenate_past_visible, self.log_crbm.delay)
            for i in range(self.log_crbm.delay-1, 0, -1):
                past_visible[i] = past_visible[i-1]
            past_visible[0] = concatenate_hidden_rbms
            concatenate_past_visible = np.concatenate(past_visible, axis=0)
            #if past_visible layers have been initialized, store the predicted label
            if(index_frame >= nb_frames_initialization):
                #print predict for this frame
                print "Frame %d : Real label %d; label found %d" %(index_frame, labelset[index_frame], predict_label)
                #add to each chunks
                for index_chunk in range(len(all_chunks_predict)):
                    all_chunks_predict[index_chunk].append(int(predict_label))
                    all_chunks_real[index_chunk].append(labelset[index_frame])
            else :
                print "[Frame %d : Real label %d; label found %d] : Ignored (First %d frames used to initialize past visibles)"\
                    %(index_frame, labelset[index_frame], predict_label,(self.log_crbm.delay*self.log_crbm.freq))
            #if a chunk is full, get the average predicted label on this chunk then delete it from lists.
            if(len(all_chunks_predict)>0 and len(all_chunks_predict[0])%chunk_size==0):
                predict_chunk = 0
                real_chunk = 0
                #get the average predict label and real label on this chunk (chunk_size frames)
                for label in range(1,self.log_crbm.n_label):
                        if(all_chunks_predict[0].count(label)>all_chunks_predict[0].count(predict_chunk)):
                            predict_chunk=label
                        if(all_chunks_real[0].count(label)>all_chunks_real[0].count(real_chunk)):
                            real_chunk=label
                print "-> Chunk : Real label %d; label found %d \n" %(real_chunk, predict_chunk)
                #actualize confusion matrix
                confusion_matrix[predict_chunk,real_chunk]+=1
                #delete this chunk from lists
                del all_chunks_predict[0]
                del all_chunks_real[0]
        #recognition is done, get execution time and return the confusion matrix
        end_time = timeit.default_timer()
        recognition_time = (end_time - start_time)
        print ('Time for recognition : %f secondes ; Duration of file : %f secondes.' %(recognition_time,number_of_frames/120.))
        return confusion_matrix

def create_train_tdbn(training_files=None, training_labels = None,
                      validation_files=None, validation_labels = None,
                      test_files=None, test_labels = None,
                      rbm_training_epoch = 10000, rbm_learning_rate=1e-3, rbm_n_hidden=30, batch_size = 1000,
                      crbm_training_epoch = 10000, crbm_learning_rate = 1e-3, crbm_n_hidden = 300, crbm_n_delay=6, crbm_freq=10,
                      finetune_epoch = 10000, finetune_learning_rate = 0.1, log_n_label=9, plot_filename=None):

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
        #for plot :
        title = "RBMs training phase : \n epoch="+str(rbm_training_epoch)+"; Number of hidden= "+str(rbm_n_hidden)+"; Learning rate="+str(rbm_learning_rate)
        plt.title(title)
        plt.ylabel("Cost")
        plt.xlabel("Epoch")
        #number of epoch is the plot abscisse
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
        #plot_name = 'plot/rbm_'+str(timeit.default_timer())+'.png'
        plot_name = plot_filename+"plot_cost_RBMs.png"
        plt.savefig(plot_name)
        print "RBMs plot is saved!"
        plt.clf() #without this line, all is plot in the same figure
        plt.close()
    #The next step is to generate a new dataset in order to train the CRBM. To do that, we realize a propagation-up of previous dataset in RBMs
    dataname = ['trained_model/dataset_train.pkl','trained_model/dataset_valid.pkl','trained_model/dataset_test.pkl']
    labelname = ['trained_model/labelset_train.pkl','trained_model/labelset_valid.pkl','trained_model/labelset_test.pkl']
    if(regenerate_crbm_dataset):
        #propagation up function using shared variable for gpu optimization
        shared_input_tab = [] #shared inputs for our 5 rbms
        for index_rbm in range(5):
            input_type = np.zeros(rbms[index_rbm].n_visible).astype(theano.config.floatX)
            shared_input_tab.append(theano.shared(input_type.astype(theano.config.floatX)))
        output_rbm_sample = []
        for index_rbm in range(5):
            output_sigmoid_value = T.nnet.sigmoid(T.dot(shared_input_tab[index_rbm], rbms[index_rbm].W) + rbms[index_rbm].hbias)
            output_rbm_sample.append(rbms[index_rbm].theano_rng.binomial(size=output_sigmoid_value.shape,n=1, p=output_sigmoid_value,dtype=theano.config.floatX))
        concatenate_hidden_rbms = T.concatenate(output_rbm_sample, axis=0)
        prop_up_func = theano.function([], concatenate_hidden_rbms)
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
            for index_frame in range(number_of_samples):
                for index_rbm in range(5):
                    shared_input_tab[index_rbm].set_value(data_rbms[index_rbm].get_value(borrow=True)[index_frame])
                    concatenate_hidden_rbms = prop_up_func()
                #add this to dataset_crbm
                dataset_crbm.append(concatenate_hidden_rbms)
                if(index_frame%(number_of_samples//100)==0):
                    print ('CRBM dataset generation %d : %d %%' %(i,((100*index_frame/number_of_samples)+1)) )

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
    #plot_name = 'plot/PER_crbm_'+str(timeit.default_timer())+'.png'
    plot_name = plot_filename+"plot_PER_CRBM.png"
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
    #store plot
    with open(plot_filename+"_rbm_cost.pkl", 'w') as f:
            cPickle.dump(cost_rbms, f)
    with open(plot_filename+"_PER_valid_x.pkl", 'w') as f:
            cPickle.dump(PER_x_valid, f)
    with open(plot_filename+"_PER_valid_y.pkl", 'w') as f:
            cPickle.dump(PER_y_valid, f)
    with open(plot_filename+"_PER_test_x.pkl", 'w') as f:
            cPickle.dump(PER_x_test, f)
    with open(plot_filename+"_PER_test_y.pkl", 'w') as f:
            cPickle.dump(PER_y_test, f)
    return tdbn, pretraining_time

def confusion_matrix_analysis(confusion_matrix=None, n_labels=None, filename=None):
    output_file = open(filename, "w")
    output_file.write("Confusion matrix :\n")
    np.savetxt(output_file,confusion_matrix)
    number_of_frames = np.sum(confusion_matrix)
    output_file.write("Number of frames (or chunks) in the test dataset : %d\n" %number_of_frames)
    output_file.write("\n->Analysis :\n")
    #for each gestures (label)
    for i in range(n_labels):
        output_file.write("Gesture label %d (gesture%d*.bvh):\n" %(i,i+1))
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
        np.savetxt(output_file,confusion_matrix_gesture)
        precision = TP/(TP+FP)
        recall = TP/(TP+FN)
        accuracy = (TP+TN)/number_of_frames
        f_measure = 2/((1/precision)+(1/recall))
        output_file.write("Nb frames : %d\n" %(TP+FP))
        output_file.write("Precision : %f %%\n" %(precision*100))
        output_file.write("Recall : %f %%\n" %(recall*100))
        output_file.write("Accuracy : %f %%\n" %(accuracy*100))
        output_file.write("F_measure : %f %%\n" %(f_measure*100))
        output_file.write("-------------\n")
    PER = 1 - np.sum(np.diag(confusion_matrix))/number_of_frames
    output_file.write("PER = %f %% (for frame by frame analysis)" %(PER*100))
    output_file.close()

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


    ###EXPERIMENTS#######
    rbm_hidden_tab=[10,30,50]
    crbm_hidden_tab=[100,300,500]
    crbm_delay_tab=[6,9,15]
    crbm_freq_tab=[3,5,10,15]
    expe=0
    info_file = open("list_expe_time.txt", "w")

    for a in range(3):
        for b in range(3):
            for c in range(3):
                for d in range(4):

                    filename = "plot/expe"+str(expe)

                    tdbn, training_time = create_train_tdbn(
                                        training_files = training_files, training_labels = training_labels,
                                        validation_files= validation_files, validation_labels = validation_labels,
                                        test_files = test_files, test_labels = test_labels,
                                        rbm_training_epoch = 10000, rbm_learning_rate=1e-3, rbm_n_hidden=rbm_hidden_tab[a], batch_size = 500,
                                        crbm_training_epoch = 10000, crbm_learning_rate = 1e-3, crbm_n_hidden =crbm_hidden_tab[b], crbm_n_delay=crbm_delay_tab[c], crbm_freq=crbm_freq_tab[d],
                                        finetune_epoch = 10000, finetune_learning_rate = 0.1, log_n_label=9, plot_filename=filename)

                    confusion_matrix = np.zeros((tdbn.log_crbm.n_label,tdbn.log_crbm.n_label)) #REAL : column, #Predict : line
                    for file_index in range(len(test_files)):
                        print "Filename : "+test_files[file_index]+ " (label %d)" %test_labels[file_index]
                        confusion_matrix += tdbn.recognize_files_chunk(test_files[file_index], test_labels[file_index])
                    confusion_name = filename+"_confusion_mat_number.txt"
                    confusion_matrix_analysis(confusion_matrix, tdbn.log_crbm.n_label, confusion_name)

                    info_file.write("Experience number ")
                    info_file.write(str(expe))
                    info_file.write("training time : ")
                    info_file.write(str(training_time/60.))
                    info_file.write(" rbm hidden : ")
                    info_file.write(str(rbm_hidden_tab[a]))
                    info_file.write(" crbm hidden : ")
                    info_file.write(str(crbm_hidden_tab[b]))
                    info_file.write(" crbm delay : ")
                    info_file.write(str(crbm_delay_tab[c]))
                    info_file.write(" crbm freq : ")
                    info_file.write(str(crbm_freq_tab[d]))
                    expe+=1
    info_file.close()

    """
    #Here, there are two possibilities :
    #You're prevously chosen to train a TDBN, so create it and save it
    if do_training_phase:
        #create and train a tdbn
        tdbn = create_train_tdbn(
                            training_files = training_files, training_labels = training_labels,
                            validation_files= validation_files, validation_labels = validation_labels,
                            test_files = test_files, test_labels = test_labels, plot_filename="plot/test")
        #save our network
        with open('trained_model/best_model_tdbn.pkl', 'w') as f:
            cPickle.dump(tdbn, f)
        print "TDBN network is saved!"

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
        confusion_matrix_analysis(confusion_matrix, tdbn.log_crbm.n_label, "plot/confusion_mat_number.txt")
    """
