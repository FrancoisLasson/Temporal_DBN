"""This class describe restricted boltzmann machines (RBM) using Theano.
Boltzmann Machines (BMs) are a particular form of energy-based model which
contain hidden variables. Restricted Boltzmann Machines further restrict BMs
to those without visible-visible and hidden-hidden connections.
@author Lisa lab : To see more http://deeplearning.net/tutorial/rbm.html

@Modification : Francois Lasson : CERV Brest France
@mail : francois.lasson@gmail.com
"""

import numpy
import theano
import theano.tensor as T
#theano Random
from theano.tensor.shared_randomstreams import RandomStreams

#Use to compute learning time
import timeit
#Use to save trained model or dataset
import cPickle
#This class used to generate dataset readable by RBMs from BVH files
from motion import generate_dataset

class RBM(object):
    """Restricted Boltzmann Machine (RBM)  """
    def __init__(self, input=None, n_visible=21, n_hidden=30,
                W=None, hbias=None, vbias=None,
                numpy_rng=None, theano_rng=None):
        """
        RBM constructor. Defines the parameters of the model along with
        basic operations for inferring hidden from visible (and vice-versa),
        as well as for performing CD updates.

        :param input: None for standalone RBMs or symbolic variable if RBM is
        part of a larger graph.

        :param n_visible: number of visible units

        :param n_hidden: number of hidden units

        :param W: None for standalone RBMs or symbolic variable pointing to a
        shared weight matrix in case RBM is part of a DBN network; in a DBN,
        the weights are shared between RBMs and layers of a MLP

        :param hbias: None for standalone RBMs or symbolic variable pointing
        to a shared hidden units bias vector in case RBM is part of a
        different network

        :param vbias: None for standalone RBMs or a symbolic variable
        pointing to a shared visible units bias
        """

        self.n_visible = n_visible
        self.n_hidden = n_hidden

        if numpy_rng is None:
            # create a number generator
            numpy_rng = numpy.random.RandomState(1234)

        if theano_rng is None:
            theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))

        if W is None:
            # W is initialized with `initial_W` which is uniformely
            # sampled from -4*sqrt(6./(n_visible+n_hidden)) and
            # 4*sqrt(6./(n_hidden+n_visible)) the output of uniform if
            # converted using asarray to dtype theano.config.floatX so
            # that the code is runable on GPU
            initial_W = numpy.asarray(
                numpy_rng.uniform(
                    low=-4 * numpy.sqrt(6. / (n_hidden + n_visible)),
                    high=4 * numpy.sqrt(6. / (n_hidden + n_visible)),
                    size=(n_visible, n_hidden)
                ),
                dtype=theano.config.floatX
            )
            # theano shared variables for weights and biases
            W = theano.shared(value=initial_W, name='W', borrow=True)

        if hbias is None:
            # create shared variable for hidden units bias
            hbias = theano.shared(
                value=numpy.zeros(
                    n_hidden,
                    dtype=theano.config.floatX
                ),
                name='hbias',
                borrow=True
            )

        if vbias is None:
            # create shared variable for visible units bias
            vbias = theano.shared(
                value=numpy.zeros(
                    n_visible,
                    dtype=theano.config.floatX
                ),
                name='vbias',
                borrow=True
            )

        # initialize input layer for standalone RBM or layer0 of DBN
        self.input = input
        if not input:
            self.input = T.matrix('input')# T.fvector('input') #float 64 bits vector

        self.W = W
        self.hbias = hbias
        self.vbias = vbias
        self.theano_rng = theano_rng
        # **** WARNING: It is not a good idea to put things in this list
        # other than shared variables created in this function.
        self.params = [self.W, self.hbias, self.vbias]

    def free_energy(self, v_sample):
        ''' Function to compute the free energy '''
        wx_b = T.dot(v_sample, self.W) + self.hbias
        vbias_term = T.dot(v_sample, self.vbias)
        hidden_term = T.sum(T.log(1 + T.exp(wx_b)), axis=1)
        return -hidden_term - vbias_term

    def propup(self, vis):
        '''This function propagates the visible units activation upwards to
        the hidden units

        Note that we return also the pre-sigmoid activation of the
        layer. As it will turn out later, due to how Theano deals with
        optimizations, this symbolic variable will be needed to write
        down a more stable computational graph (see details in the
        reconstruction cost function)

        '''
        pre_sigmoid_activation = T.dot(vis, self.W) + self.hbias
        return [pre_sigmoid_activation, T.nnet.sigmoid(pre_sigmoid_activation)]

    def sample_h_given_v(self, v0_sample):
        ''' This function infers state of hidden units given visible units '''
        # compute the activation of the hidden units given a sample of
        # the visibles
        pre_sigmoid_h1, h1_mean = self.propup(v0_sample)
        # get a sample of the hiddens given their activation
        # Note that theano_rng.binomial returns a symbolic sample of dtype
        # int64 by default. If we want to keep our computations in floatX
        # for the GPU we need to specify to return the dtype floatX
        h1_sample = self.theano_rng.binomial(size=h1_mean.shape,
                                             n=1, p=h1_mean,
                                             dtype=theano.config.floatX)
        return [pre_sigmoid_h1, h1_mean, h1_sample]

    def propdown(self, hid):
        '''This function propagates the hidden units activation downwards to
        the visible units

        Note that we return also the pre_sigmoid_activation of the
        layer. As it will turn out later, due to how Theano deals with
        optimizations, this symbolic variable will be needed to write
        down a more stable computational graph (see details in the
        reconstruction cost function)

        '''
        pre_sigmoid_activation = T.dot(hid, self.W.T) + self.vbias
        return [pre_sigmoid_activation, T.nnet.sigmoid(pre_sigmoid_activation)]

    def sample_v_given_h(self, h0_sample):
        ''' This function infers state of visible units given hidden units '''
        # compute the activation of the visible given the hidden sample
        pre_sigmoid_v1, v1_mean = self.propdown(h0_sample)
        # get a sample of the visible given their activation
        # Note that theano_rng.binomial returns a symbolic sample of dtype
        # int64 by default. If we want to keep our computations in floatX
        # for the GPU we need to specify to return the dtype floatX
        v1_sample = self.theano_rng.binomial(size=v1_mean.shape,
                                             n=1, p=v1_mean,
                                             dtype=theano.config.floatX)
        return [pre_sigmoid_v1, v1_mean, v1_sample]

    def gibbs_hvh(self, h0_sample):
        ''' This function implements one step of Gibbs sampling,
            starting from the hidden state'''
        pre_sigmoid_v1, v1_mean, v1_sample = self.sample_v_given_h(h0_sample)
        pre_sigmoid_h1, h1_mean, h1_sample = self.sample_h_given_v(v1_sample)
        return [pre_sigmoid_v1, v1_mean, v1_sample,
                pre_sigmoid_h1, h1_mean, h1_sample]

    def gibbs_vhv(self, v0_sample):
        ''' This function implements one step of Gibbs sampling,
            starting from the visible state'''
        pre_sigmoid_h1, h1_mean, h1_sample = self.sample_h_given_v(v0_sample)
        pre_sigmoid_v1, v1_mean, v1_sample = self.sample_v_given_h(h1_sample)
        return [pre_sigmoid_h1, h1_mean, h1_sample,
                pre_sigmoid_v1, v1_mean, v1_sample]

    def get_cost_updates(self, lr=0.1, persistent=None, k=1):
        """This functions implements one step of CD-k or PCD-k

        :param lr: learning rate used to train the RBM

        :param persistent: None for CD. For PCD, shared variable
            containing old state of Gibbs chain. This must be a shared
            variable of size (batch size, number of hidden units).

        :param k: number of Gibbs steps to do in CD-k/PCD-k

        Returns a proxy for the cost and the updates dictionary. The
        dictionary contains the update rules for weights and biases but
        also an update of the shared variable used to store the persistent
        chain, if one is used.

        """

        # compute positive phase
        pre_sigmoid_ph, ph_mean, ph_sample = self.sample_h_given_v(self.input)

        # decide how to initialize persistent chain:
        # for CD, we use the newly generate hidden sample
        # for PCD, we initialize from the old state of the chain
        if persistent is None:
            chain_start = ph_sample
        else:
            chain_start = persistent
        # perform actual negative phase
        # in order to implement CD-k/PCD-k we need to scan over the
        # function that implements one gibbs step k times.
        # Read Theano tutorial on scan for more information :
        # http://deeplearning.net/software/theano/library/scan.html
        # the scan will return the entire Gibbs chain
        (
            [
                pre_sigmoid_nvs,
                nv_means,
                nv_samples,
                pre_sigmoid_nhs,
                nh_means,
                nh_samples
            ],
            updates
        ) = theano.scan(
            self.gibbs_hvh,
            # the None are place holders, saying that
            # chain_start is the initial state corresponding to the
            # 6th output
            outputs_info=[None, None, None, None, None, chain_start],
            n_steps=k
        )
        # determine gradients on RBM parameters
        # note that we only need the sample at the end of the chain
        chain_end = nv_samples[-1]

        cost = T.mean(self.free_energy(self.input)) - T.mean(
            self.free_energy(chain_end))
        # We must not compute the gradient through the gibbs sampling
        gparams = T.grad(cost, self.params, consider_constant=[chain_end])

        # constructs the update dictionary
        for gparam, param in zip(gparams, self.params):
            # make sure that the learning rate is of the right dtype
            updates[param] = param - gparam * T.cast(
                lr,
                dtype=theano.config.floatX
            )
        if persistent:
            # Note that this works only if persistent is a shared variable
            updates[persistent] = nh_samples[-1]
            # pseudo-likelihood is a better proxy for PCD
            monitoring_cost = self.get_pseudo_likelihood_cost(updates)
        else:
            # reconstruction cross-entropy is a better proxy for CD
            monitoring_cost = self.get_reconstruction_cost(updates,
                                                           pre_sigmoid_nvs[-1])

        return monitoring_cost, updates

    def get_pseudo_likelihood_cost(self, updates):
        """Stochastic approximation to the pseudo-likelihood"""
        # index of bit i in expression p(x_i | x_{\i})
        bit_i_idx = theano.shared(value=0, name='bit_i_idx')

        # binarize the input image by rounding to nearest integer
        xi = T.round(self.input)

        # calculate free energy for the given bit configuration
        fe_xi = self.free_energy(xi)

        # flip bit x_i of matrix xi and preserve all other bits x_{\i}
        # Equivalent to xi[:,bit_i_idx] = 1-xi[:, bit_i_idx], but assigns
        # the result to xi_flip, instead of working in place on xi.
        xi_flip = T.set_subtensor(xi[:, bit_i_idx], 1 - xi[:, bit_i_idx])

        # calculate free energy with bit flipped
        fe_xi_flip = self.free_energy(xi_flip)

        # equivalent to e^(-FE(x_i)) / (e^(-FE(x_i)) + e^(-FE(x_{\i})))
        cost = T.mean(self.n_visible * T.log(T.nnet.sigmoid(fe_xi_flip -
                                                            fe_xi)))

        # increment bit_i_idx % number as part of updates
        updates[bit_i_idx] = (bit_i_idx + 1) % self.n_visible

        return cost

    def get_reconstruction_cost(self, updates, pre_sigmoid_nv):
        """Approximation to the reconstruction error

        Note that this function requires the pre-sigmoid activation as
        input.  To understand why this is so you need to understand a
        bit about how Theano works. Whenever you compile a Theano
        function, the computational graph that you pass as input gets
        optimized for speed and stability.  This is done by changing
        several parts of the subgraphs with others.  One such
        optimization expresses terms of the form log(sigmoid(x)) in
        terms of softplus.  We need this optimization for the
        cross-entropy since sigmoid of numbers larger than 30. (or
        even less then that) turn to 1. and numbers smaller than
        -30. turn to 0 which in terms will force theano to compute
        log(0) and therefore we will get either -inf or NaN as
        cost. If the value is expressed in terms of softplus we do not
        get this undesirable behaviour. This optimization usually
        works fine, but here we have a special case. The sigmoid is
        applied inside the scan op, while the log is
        outside. Therefore Theano will only see log(scan(..)) instead
        of log(sigmoid(..)) and will not apply the wanted
        optimization. We can not go and replace the sigmoid in scan
        with something else also, because this only needs to be done
        on the last step. Therefore the easiest and more efficient way
        is to get also the pre-sigmoid activation as an output of
        scan, and apply both the log and sigmoid outside scan such
        that Theano can catch and optimize the expression.

        """

        cross_entropy = T.mean(
            T.sum(
                self.input * T.log(T.nnet.sigmoid(pre_sigmoid_nv)) +
                (1 - self.input) * T.log(1 - T.nnet.sigmoid(pre_sigmoid_nv)),
                axis=1
            )
        )

        return cross_entropy

    def predict_hidden(self, frame=None) :
        #declare theano function
        visible = T.vector('input')
        [pre_sigmoid_hidden, hidden, hidden_sample] = self.sample_h_given_v(visible)
        f = theano.function([visible], hidden_sample) #sample with a binomial
        return f(frame)


def create_train_rbm(learning_rate=1e-3, training_epochs=5000,
             dataset=None, seqlen = None, batch_size=20, n_hidden=30):
    """
    Demonstrate how to train a RBM
    :param learning_rate: learning rate used for training the RBM
    :param training_epochs: number of epochs used for training
    :param dataset: path the the pickled dataset
    :param batch_size: size of a batch used to train the RBM
    :param n_chains: number of parallel Gibbs chains to be used for sampling
    :param n_samples: number of samples to plot for each chain
    """
    #for random generation :seed
    rng = numpy.random.RandomState(123)
    theano_rng = RandomStreams(rng.randint(2 ** 30))
    # compute number of minibatches for training, validation and testing
    n_train_batches = dataset.get_value(borrow=True).shape[0] / batch_size #number of minibatch
    n_dim = dataset.get_value(borrow=True).shape[1] #number of data in each frames

    # allocate symbolic variables for the data
    index = T.lvector() # list of index : shuffle data
    x = T.matrix('x')  # the data : matrix cos to minibatch?

    # initialize storage for the persistent chain (state = hidden
    # layer of chain)
    persistent_chain = theano.shared(numpy.zeros((batch_size, n_hidden), dtype=theano.config.floatX), borrow=True)

    # construct the RBM class
    rbm = RBM(input=x, n_visible=n_dim, n_hidden=n_hidden, numpy_rng=rng, theano_rng=theano_rng)

    # get the cost and the gradient corresponding to one step of CD-15
    cost, updates = rbm.get_cost_updates(lr=learning_rate, persistent=persistent_chain, k=1)

    #################################
    #     Training the RBM          #
    #################################

    # it is ok for a theano function to have no output
    # the purpose of train_rbm is solely to update the RBM parameters
    train_rbm = theano.function(
        [index], #minibatch index
        cost,
        updates=updates,
        givens={ x: dataset[index]}, #for the [index]minibatch
        name='train_rbm'
    )

    plotting_time = 0.
    start_time = timeit.default_timer()
    #shuffle data : Learn gesture after gesture could introduce a bias. In order to avoid this,
    #we shuffle data to learn all gestures at the same time
    datasetindex = []
    last = 0
    for s in seqlen:
        datasetindex += range(last, last + s)
        last += s
    permindex = numpy.array(datasetindex)
    rng.shuffle(permindex)

    #In order visualize cost evolution during training phase
    cost_y = []
    # go through training epochs
    for epoch in xrange(training_epochs):
        # go through the training set
        mean_cost = []
        for batch_index in xrange(n_train_batches): #for each minibatch
            data_idx = permindex[batch_index * batch_size:(batch_index + 1) * batch_size] #get a list of index in the shuffle index-list
            mean_cost += [train_rbm(data_idx)]
        print 'Training epoch %d, cost is ' % epoch, numpy.mean(mean_cost)
        cost_y.append(numpy.mean(mean_cost))

    end_time = timeit.default_timer()
    pretraining_time = (end_time - start_time) - plotting_time
    print ('RBM : Training took %f minutes' % (pretraining_time / 60.))
    return rbm, cost_y

"""
def create_train_rbm_GPU(learning_rate=1e-3, training_epochs=5000,
             dataset=None, seqlen = None, batch_size=20, n_hidden=30):
    #for random generation :seed
    rng = numpy.random.RandomState(123)
    theano_rng = RandomStreams(rng.randint(2 ** 30))
    # compute number of minibatches for training, validation and testing
    n_train_batches = dataset.get_value(borrow=True).shape[0] / batch_size #number of minibatch
    n_dim = dataset.get_value(borrow=True).shape[1] #number of data in each frames

    # allocate shared variables for the data : gpu optimization
    x_type = numpy.zeros((n_dim,batch_size)).astype(theano.config.floatX)
    shared_x = theano.shared(x_type.astype(theano.config.floatX))
    h_type = numpy.zeros((n_hidden,batch_size)).astype(theano.config.floatX)
    shared_h = theano.shared(h_type.astype(theano.config.floatX))

    # initialize storage for the persistent chain (state = hidden layer of chain)
    persistent_chain = theano.shared(numpy.zeros((batch_size, n_hidden), dtype=theano.config.floatX), borrow=True)
    # construct the RBM class
    rbm = RBM(input=shared_x, n_visible=n_dim, n_hidden=n_hidden, numpy_rng=rng, theano_rng=theano_rng)

    #prop up and sample visible
    h_pre = T.dot(shared_x, rbm.W) + rbm.hbias
    h_mean = T.nnet.sigmoid(h_pre)
    h_sample = theano_rng.binomial(size=h_mean.shape, n=1, p=h_mean, dtype=theano.config.floatX)
    propup_sample = theano.function([], [h_pre, h_mean, h_sample])
    #prop up and sample visible


    v_pre = T.dot(shared_h, rbm.W) + rbm.vbias
    v_mean = T.nnet.sigmoid(v_pre)
    v_sample = theano_rng.binomial(size=v_mean.shape, n=1, p=v_mean, dtype=theano.config.floatX)
    propdown_sample = theano.function([], [v_pre, v_mean, v_sample])
    #gibbs hvh
    print "ok\n\n\n\n"
    v_pre_2,v_mean_2,v_sample_2=propdown_sample()
    h_pre_2, h_mean_2, h_sample_2=propup_sample()
    gibbs_hvh = theano.function([], [v_pre_2, v_mean_2, v_sample_2, h_pre_2, h_mean_2, h_sample_2])

    print "ok\n\n\n\n"
    # compute positive phase
    shared_h.set_value(propup_sample()[2])


    ([pre_sigmoid_nvs, nv_means, nv_samples, pre_sigmoid_nhs, nh_means, nh_samples], updates) = theano.scan(
        rbm.gibbs_hvh, outputs_info=[None, None, None, None, None, persistent_chain], n_steps=1)
    # determine gradients on RBM parameters
    # note that we only need the sample at the end of the chain
    chain_end = nv_samples[-1]
    cost = T.mean(rbm.free_energy(shared_x)) - T.mean(rbm.free_energy(chain_end))
    # We must not compute the gradient through the gibbs sampling
    gparams = T.grad(cost, rbm.params, consider_constant=[chain_end])
    # constructs the update dictionary
    for gparam, param in zip(gparams, rbm.params):
        # make sure that the learning rate is of the right dtype
        updates[param] = param - gparam * T.cast(learning_rate,dtype=theano.config.floatX)

    # Note that this works only if persistent is a shared variable
    updates[persistent_chain] = nh_samples[-1]
    # pseudo-likelihood is a better proxy for PCD
    bit_i_idx = theano.shared(value=0, name='bit_i_idx')
    # binarize the input image by rounding to nearest integer
    xi = T.round(shared_x)
    # calculate free energy for the given bit configuration
    fe_xi = rbm.free_energy(xi)
    # flip bit x_i of matrix xi and preserve all other bits x_{\i}
    # Equivalent to xi[:,bit_i_idx] = 1-xi[:, bit_i_idx], but assigns
    # the result to xi_flip, instead of working in place on xi.
    xi_flip = T.set_subtensor(xi[:, bit_i_idx], 1 - xi[:, bit_i_idx])
    # calculate free energy with bit flipped
    fe_xi_flip = rbm.free_energy(xi_flip)
    # equivalent to e^(-FE(x_i)) / (e^(-FE(x_i)) + e^(-FE(x_{\i})))
    cost = T.mean(rbm.n_visible * T.log(T.nnet.sigmoid(fe_xi_flip -fe_xi)))
    # increment bit_i_idx % number as part of updates
    updates[bit_i_idx] = (bit_i_idx + 1) % rbm.n_visible

    train_rbm = theano.function([], cost, updates = updates)


    plotting_time = 0.
    start_time = timeit.default_timer()
    #shuffle data : Learn gesture after gesture could introduce a bias. In order to avoid this,
    #we shuffle data to learn all gestures at the same time
    datasetindex = []
    last = 0
    for s in seqlen:
        datasetindex += range(last, last + s)
        last += s
    permindex = numpy.array(datasetindex)
    rng.shuffle(permindex)

    #In order visualize cost evolution during training phase
    cost_y = []
    # go through training epochs
    for epoch in xrange(training_epochs):
        # go through the training set
        mean_cost = []
        for batch_index in xrange(n_train_batches): #for each minibatch
            data_idx = permindex[batch_index * batch_size:(batch_index + 1) * batch_size] #get a list of index in the shuffle index-list
            #mean_cost += [train_rbm(data_idx)] #TODO



            shared_x.set_value(dataset.get_value()[data_idx])
            mean_cost += [train_rbm()]









            ##########################################
        print 'Training epoch %d, cost is ' % epoch, numpy.mean(mean_cost)
        cost_y.append(numpy.mean(mean_cost))
    end_time = timeit.default_timer()
    pretraining_time = (end_time - start_time) - plotting_time
    print ('RBM : Training took %f minutes' % (pretraining_time / 60.))
    cost_y=0
    return rbm, cost_y
"""

if __name__ == '__main__':
    #This main allow us to test our RBM on bvh dataset
    #Firstly, do you want to train a RBM or do you want to test it
    do_training_phase = True

    #To create a RBM, we have to train our model from training files
    #To do that, we have to define BVH files use to create datasets, i.e. a training set
    training_files = ['data/geste1i.bvh',
                  'data/geste2i.bvh',
                  'data/geste3i.bvh',
                  'data/geste13i.bvh',
                  'data/geste5i.bvh',
                  'data/geste14i.bvh',
                  'data/geste18i.bvh',
                  'data/geste8i.bvh',
                  'data/geste10i.bvh']
    training_labels = [0,1,2,3,4,5,6,7,8]
    #Here, there are two possibilities :
    #You're prevously chosen to train a RBM, so create it and save it
    if do_training_phase:
        trainingSet, ignored, ignored, ignored, ignored, ignored, trainingLen = generate_dataset(training_files, training_labels)
        rbm = create_train_rbm(dataset=trainingSet, seqlen = trainingLen)[0]
        #save our network
        with open('best_model_rbm.pkl', 'w') as f:
            cPickle.dump(rbm, f)

    #Else, you have already trained a RBM and you want to test it
    else :
        #load our Network
        rbm = cPickle.load(open('best_model_rbm.pkl'))
        #file and real label
        files = ['data/geste8i.bvh'] #Mocap file used to test RBM
        labels = [7] #Useless but must to be due to generate_dataset function

        #predict frame after frame : function
        #input RBM var
        frame = numpy.ones(rbm.n_visible).astype(theano.config.floatX)
        input = theano.shared(frame.astype(theano.config.floatX), name='input')
        #function to propup
        output_sigmoid_value = T.nnet.sigmoid(T.dot(input, rbm.W) + rbm.hbias)
        output_sample = rbm.theano_rng.binomial(size=output_sigmoid_value.shape,
                                                n=1, p=output_sigmoid_value,
                                                dtype=theano.config.floatX)
        prop_up_func = theano.function([], output_sample)

        data = generate_dataset(files, labels)[0]
        number_of_frames = data.get_value(borrow=True).shape[0]
        for index_frame in range(number_of_frames):
            frame_in = data.get_value(borrow=True)[index_frame]
            input.set_value(frame_in)
            print "Frame %d: " %index_frame
            #print frame_in
            print "Get hidden label : "
            print prop_up_func()
            #print rbm.predict_hidden(frame_in)
            print "--------------------"
