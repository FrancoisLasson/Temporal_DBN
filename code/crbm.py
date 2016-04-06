"""
This class implement a Conditional RBM, a temporal generative model
To understand this model : Taylor http://www.uoguelph.ca/~gwtaylor/publications/nips2006mhmublv/
Taylor Github : https://gist.github.com/gwtaylor/2505670 (CRBM theano implementation without modification)
@author Graham W. Taylor

@modification Francois Lasson : CERV Brest France
@mail : francois.lasson@gmail.com
"""

#Numpy and Theano library
import numpy as np
import theano
import theano.tensor as T
#theano Random
from theano.tensor.shared_randomstreams import RandomStreams

#Use to compute learning time
import timeit

#This class used to generate dataset readable by RBMs from BVH files
from motion import generate_dataset #to load mocap


class CRBM(object):
    """Conditional Restricted Boltzmann Machine (CRBM)  """
    def __init__(self, input=None, input_history=None,
                n_visible=49, n_hidden=15, delay=6,
                A=None, B=None, W=None, hbias=None, vbias=None,
                numpy_rng=None, theano_rng=None):
        """
        CRBM constructor. Defines the parameters of the model along with
        basic operations for inferring hidden from visible (and vice-versa),
        as well as for performing CD updates.

        :param input: None for standalone RBMs or symbolic variable if RBM is
        part of a larger graph.

        :param n_visible: number of visible units

        :param n_hidden: number of hidden units

        :param A: None for standalone CRBMs or symbolic variable pointing to a
        shared weight matrix in case CRBM is part of a CDBN network; in a CDBN,
        the weights are shared between CRBMs and layers of a MLP

        :param B: None for standalone CRBMs or symbolic variable pointing to a
        shared weight matrix in case CRBM is part of a CDBN network; in a CDBN,
        the weights are shared between CRBMs and layers of a MLP

        :param W: None for standalone CRBMs or symbolic variable pointing to a
        shared weight matrix in case CRBM is part of a CDBN network; in a CDBN,
        the weights are shared between CRBMs and layers of a MLP

        :param hbias: None for standalone CRBMs or symbolic variable pointing
        to a shared hidden units bias vector in case CRBM is part of a
        different network

        :param vbias: None for standalone RBMs or a symbolic variable
        pointing to a shared visible units bias
        """

        #Initialize architecture (number of layers, of neurones)
        self.n_visible = n_visible #number of neurones in visible layer
        self.n_hidden = n_hidden #number of neurones in hidden layer
        self.delay = delay #number of past visible layers

        #Initialize seed for random
        if numpy_rng is None:
            # create a number generator
            numpy_rng = np.random.RandomState(1234)
        if theano_rng is None:
            theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))

        #Initialize weight matrices and biases
        if W is None:
            # the output of uniform if converted using asarray to dtype
            # theano.config.floatX so that the code is runable on GPU
            initial_W = np.asarray(0.01 * numpy_rng.randn(n_visible,
                                                          n_hidden),
                                    dtype=theano.config.floatX)
            # theano shared variables for weights and biases
            W = theano.shared(value=initial_W, name='W')
        if A is None:
            initial_A = np.asarray(0.01 * numpy_rng.randn(n_visible * delay,
                                                           n_visible),
                                    dtype=theano.config.floatX)
            # theano shared variables for weights and biases
            A = theano.shared(value=initial_A, name='A')
        if B is None:
            initial_B = np.asarray(0.01 * numpy_rng.randn(n_visible * delay,
                                                           n_hidden),
                                    dtype=theano.config.floatX)
            # theano shared variables for weights and biases
            B = theano.shared(value=initial_B, name='B')
        if hbias is None:
            # create shared variable for hidden units bias
            hbias = theano.shared(value=np.zeros(n_hidden,
                                dtype=theano.config.floatX), name='hbias')
        if vbias is None:
            # create shared variable for visible units bias
            vbias = theano.shared(value=np.zeros(n_visible,
                                dtype=theano.config.floatX), name='vbias')

        # initialize input layer for standalone CRBM or layer0 of CDBN
        self.input = input
        if not input:
            self.input = T.matrix('input')
        self.input_history = input_history
        if not input_history:
            self.input_history = T.matrix('input_history')

        self.W = W
        self.A = A
        self.B = B
        self.hbias = hbias
        self.vbias = vbias
        self.theano_rng = theano_rng
        self.numpy_rng = numpy_rng
        self.params = [self.W, self.A, self.B, self.hbias, self.vbias]

    def free_energy(self, v_sample, v_history):
        ''' Function to compute the free energy of a sample conditional
        on the history '''
        wx_b = T.dot(v_sample, self.W) + T.dot(v_history, self.B) + self.hbias
        ax_b = T.dot(v_history, self.A) + self.vbias
        visible_term = T.sum(0.5 * T.sqr(v_sample - ax_b), axis=1)
        hidden_term = T.sum(T.log(1 + T.exp(wx_b)), axis=1)
        return visible_term - hidden_term #TODO Useless, this Energy Function allow net to work with continuous value, however we use binary value

    def propup(self, vis, v_history):
        ''' This function propagates the visible units activation upwards to
        the hidden units

        Note that we return also the pre-sigmoid activation of the layer. As
        it will turn out later, due to how Theano deals with optimizations,
        this symbolic variable will be needed to write down a more
        stable computational graph (see details in the reconstruction cost
        function)
        '''
        pre_sigmoid_activation = T.dot(vis, self.W) + T.dot(v_history, self.B) + self.hbias
        return [pre_sigmoid_activation, T.nnet.sigmoid(pre_sigmoid_activation)]

    def sample_h_given_v(self, v0_sample, v_history):
        ''' This function infers state of hidden units given visible units '''
        # compute the activation of the hidden units given a sample of the
        # visibles
        #pre_sigmoid_h1, h1_mean = self.propup(v0_sample)
        pre_sigmoid_h1, h1_mean = self.propup(v0_sample, v_history)
        # get a sample of the hiddens given their activation
        # Note that theano_rng.binomial returns a symbolic sample of dtype
        # int64 by default. If we want to keep our computations in floatX
        # for the GPU we need to specify to return the dtype floatX
        h1_sample = self.theano_rng.binomial(size=h1_mean.shape, n=1,
                                             p=h1_mean,
                                             dtype=theano.config.floatX)
        return [pre_sigmoid_h1, h1_mean, h1_sample]

    def propdown(self, hid, v_history):
        '''This function propagates the hidden units activation downwards to
        the visible units

        Note that we return also the pre_sigmoid_activation of the layer. As
        it will turn out later, due to how Theano deals with optimizations,
        this symbolic variable will be needed to write down a more
        stable computational graph (see details in the reconstruction cost
        function)
        '''
        pre_sigmoid_activation = T.dot(hid, self.W.T) + T.dot(v_history, self.A) + self.vbias
        return [pre_sigmoid_activation, T.nnet.sigmoid(pre_sigmoid_activation)]

    def sample_v_given_h(self, h0_sample, v_history):
        ''' This function infers state of visible units given hidden units '''
        # compute the activation of the visible given the hidden sample
        #pre_sigmoid_v1, v1_mean = self.propdown(h0_sample)
        pre_sigmoid_v1, v1_mean = self.propdown(h0_sample, v_history)
        # get a sample of the visible given their activation
        # Note that theano_rng.binomial returns a symbolic sample of dtype
        # int64 by default. If we want to keep our computations in floatX
        # for the GPU we need to specify to return the dtype floatX
        v1_sample = self.theano_rng.binomial(size=v1_mean.shape,
                                             n=1, p=v1_mean,
                                             dtype = theano.config.floatX)
        return [pre_sigmoid_v1, v1_mean, v1_sample]

    def gibbs_hvh(self, h0_sample, v_history):
        ''' This function implements one step of Gibbs sampling,
            starting from the hidden state'''
        pre_sigmoid_v1, v1_mean, v1_sample = self.sample_v_given_h(h0_sample, v_history)
        pre_sigmoid_h1, h1_mean, h1_sample = self.sample_h_given_v(v1_sample, v_history)
        return [pre_sigmoid_v1, v1_mean, v1_sample, pre_sigmoid_h1, h1_mean, h1_sample]

    def gibbs_vhv(self, v0_sample, v_history):
        ''' This function implements one step of Gibbs sampling,
            starting from the visible state'''
        pre_sigmoid_h1, h1_mean, h1_sample = self.sample_h_given_v(v0_sample, v_history)
        pre_sigmoid_v1, v1_mean, v1_sample = self.sample_v_given_h(h1_sample, v_history)
        return [pre_sigmoid_h1, h1_mean, h1_sample, pre_sigmoid_v1, v1_mean, v1_sample]

    def get_cost_updates(self, lr=0.1, k=1):
        """
        This functions implements one step of CD-k
        :param lr: learning rate used to train the RBM
        :param persistent: None for CD
        :param k: number of Gibbs steps to do in CD-k
        Returns a proxy for the cost and the updates dictionary. The
        dictionary contains the update rules for weights and biases but
        also an update of the shared variable used to store the persistent
        chain, if one is used.
        """
        # compute positive phase
        pre_sigmoid_ph, ph_mean, ph_sample = self.sample_h_given_v(self.input, self.input_history)

        # for CD, we use the newly generate hidden sample
        chain_start = ph_sample

        # perform actual negative phase
        # in order to implement CD-k we need to scan over the
        # function that implements one gibbs step k times.
        # Read Theano tutorial on scan for more information :
        # http://deeplearning.net/software/theano/library/scan.html
        # the scan will return the entire Gibbs chain
        # updates dictionary is important because it contains the updates
        # for the random number generator
        [pre_sigmoid_nvs, nv_means, nv_samples, pre_sigmoid_nhs, nh_means,
         nh_samples], updates = theano.scan(self.gibbs_hvh,
                    # the None are place holders, saying that
                    # chain_start is the initial state corresponding to the
                    # 5th output
                    outputs_info=[None, None, None, None, None, chain_start],
                    non_sequences=self.input_history,
                    n_steps=k)

        # determine gradients on CRBM parameters
        # not that we only need the sample at the end of the chain
        chain_end = nv_samples[-1]

        cost = T.mean(self.free_energy(self.input, self.input_history)) - T.mean(self.free_energy(chain_end, self.input_history))
        # We must not compute the gradient through the gibbs sampling
        gparams = T.grad(cost, self.params, consider_constant=[chain_end]) #consider_constant=[chain_end]

        # constructs the update dictionary
        for gparam, param in zip(gparams, self.params):
            # make sure that the learning rate is of the right dtype
            if param == self.A:
                # slow down autoregressive updates
                updates[param] = param - gparam * 0.01 * \
                                 T.cast(lr, dtype=theano.config.floatX)
            else:
                updates[param] = param - gparam * \
                                 T.cast(lr, dtype=theano.config.floatX)

        # reconstruction error is a better proxy for CD
        monitoring_cost = self.get_reconstruction_cost(updates, nv_means[-1])
        return monitoring_cost, updates

    def get_reconstruction_cost(self, updates, pre_sigmoid_nv):
        """Approximation to the reconstruction error
        """
        # sum over dimensions, mean over cases
        recon = T.mean(T.sum(T.sqr(self.input - pre_sigmoid_nv), axis=1))
        return recon


    def train(self, learning_rate=None, training_epochs=None,
                 dataset=None, seqlen= None, batch_size=None):
        # compute number of minibatches for training, validation and testing
        #for each file, n_delay frames are used to initialized memory. So, this 6 frames do not have to be count for frame minibatch division
        n_train_batches = (dataset.get_value(borrow=True).shape[0]-self.delay*len(seqlen)) / batch_size

        # valid starting indices
        datasetindex = []
        last = 0
        for s in seqlen:
            datasetindex += range(last + self.delay, last + s)
            last += s
        permindex = np.array(datasetindex)
        self.numpy_rng.shuffle(permindex)

        # allocate symbolic variables for the data
        index = T.lvector()    # index to a [mini]batch
        index_hist = T.lvector()  # index to history

        # get the cost and the gradient corresponding to one step of CD-15
        cost, updates = self.get_cost_updates(lr=learning_rate, k=1)

        #################################
        #     Training the CRBM         #
        #################################
        # the purpose of train_crbm is solely to update the CRBM parameters
        train_crbm = theano.function(
                [index, index_hist],
                cost,
                updates=updates,
                givens={self.input: dataset[index],
                        self.input_history: dataset[index_hist].reshape((batch_size, self.delay * self.n_visible))
                       },
                name='train_crbm')

        start_time = timeit.default_timer()

        # go through training epochs
        for epoch in xrange(training_epochs):

            # go through the training set
            mean_cost = []
            for batch_index in xrange(n_train_batches):

                # indexing is slightly complicated
                # build a linear index to the starting frames for this batch
                # (i.e. time t) gives a batch_size length array for data
                data_idx = permindex[batch_index * batch_size:(batch_index + 1) * batch_size]
                # now build a linear index to the frames at each delay tap
                # (i.e. time t-1 to t-delay)
                # gives a batch_size x delay array of indices for history
                hist_idx = np.array([data_idx - n for n in xrange(1, self.delay + 1)]).T
                this_cost = train_crbm(data_idx, hist_idx.ravel())
                #print batch_index, this_cost
                mean_cost += [this_cost]
            print 'Training epoch %d, cost is ' % epoch, np.mean(mean_cost)
        end_time = timeit.default_timer()
        pretraining_time = (end_time - start_time)
        print ('Training took %f minutes' % (pretraining_time / 60.))


    def predict_hidden(self, dataset=None, batch_size=100):
        # compute number of minibatches for training, validation and testing
        n_test_batches = dataset.get_value(borrow=True).shape[0] / batch_size
        n_dim =  dataset.get_value(borrow=True).shape[1]

        # allocate symbolic variables for the data
        index = T.lvector()    # index to a [mini]batch
        index_hist = T.lvector()  # index to history

        #print hidden layer
        [pre_sigmoid_h1, h1_mean, h1_sample] = self.sample_h_given_v(self.input, self.input_history)

        print_hidden = theano.function(
            [index, index_hist],
            [h1_sample],
            givens={ self.input:dataset[index],
                     self.input_history:dataset[index_hist].reshape((batch_size, self.delay * self.n_visible))
                    },
            name='print_hidden'
            )

        # valid starting indices
        datasetindex = range(self.delay, dataset.get_value(borrow=True).shape[0])
        permindex = np.array(datasetindex)

        #For each frame in minibatch
        for batch_index in xrange(n_test_batches):
            data_idx = permindex[batch_index * batch_size : (batch_index + 1) * batch_size]
            hist_idx = np.array([data_idx - n for n in xrange(1, self.delay + 1)]).T
            for index_in_batch in range(batch_size) :
                print "Hidden CRBM (frame %d):" %(batch_index*batch_size+index_in_batch+1)
                print print_hidden(data_idx, hist_idx.ravel())[0][index_in_batch]
                print "-----------"


def create_train_crbm(learning_rate=1e-3, training_epochs=10000,
             dataset=None, seqlen= None, batch_size=100,
             n_hidden=50, delay=6):
    """
    Demonstrate how to train a CRBM.
    This is demonstrated on mocap data.
    :param learning_rate: learning rate used for training the CRBM
    :param training_epochs: number of epochs used for training
    :param dataset: path the the dataset (matlab format)
    :param batch_size: size of a batch used to train the RBM
    """

    rng = np.random.RandomState(123)
    theano_rng = RandomStreams(rng.randint(2 ** 30))

    n_dim =  dataset.get_value(borrow=True).shape[1]

    x = T.matrix('x')  # the data
    x_history = T.matrix('x_history')

    # construct the CRBM class
    crbm = CRBM(input=x, input_history=x_history, n_visible=n_dim, n_hidden=n_hidden, delay=delay,
                numpy_rng=rng, theano_rng=theano_rng)

    crbm.train(learning_rate=learning_rate, training_epochs=training_epochs, dataset=dataset, seqlen= seqlen, batch_size=batch_size)
    return crbm

if __name__ == '__main__':
    crbm, dataset = create_train_crbm(n_hidden=10)
