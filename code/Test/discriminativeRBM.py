"""This class describe Classification restricted boltzmann machines (RBM) using Theano.
For this kind of networks, data have to be labelised
To understand this model, you could read this paper :
Jerome LOURADOUR & Hugo LAROCHELLE _ Classification of sets using Restricted Boltzmann Machines _ page2
Author of RBM class : Lisa lab
Modification for Classification RBM : Francois Lasson, CERV Brest
"""

import timeit

try:
    import PIL.Image as Image
except ImportError:
    import Image

import numpy

import theano
import theano.tensor as T
import os

from theano.tensor.shared_randomstreams import RandomStreams

from utils import tile_raster_images
from logistic_sgd import load_data

import cPickle #to save and reload our network

# start-snippet-1
class DRBM(object):
    """Discriminative Restricted Boltzmann Machine (DRBM)  """
    def __init__(self, input=None, label=None,
            n_visible=10, n_label = 10, n_hidden=30,
            W=None, U=None, hbias=None, lbias=None, vbias=None,
            numpy_rng=None, theano_rng=None):
        """
        DRBM constructor. Defines the parameters of the model along with
        basic operations for inferring hidden from visible (and vice-versa),
        as well as for performing CD updates.

        :param input: None for standalone RBMs or symbolic variable if RBM is
        part of a larger graph.
        :param label: None for standalone RBMs or symbolic variable if RBM is
        part of a larger graph.

        :param n_visible: number of visible units
        :param n_label: number of label units
        :param n_hidden: number of hidden units

        :param W: None for standalone RBMs or symbolic variable pointing to a
        shared weight matrix in case RBM is part of a DBN network; in a DBN,
        the weights are shared between RBMs and layers of a MLP
        :param U: None for standalone RBMs or symbolic variable pointing to a
        shared weight matrix of label layer

        :param hbias: None for standalone RBMs or symbolic variable pointing
        to a shared hidden units bias vector in case RBM is part of a
        different network
        :param lbias: None for standalone RBMs or a symbolic variable
        pointing to a shared label units bias
        :param vbias: None for standalone RBMs or a symbolic variable
        pointing to a shared visible units bias
        """

        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.n_label = n_label

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

        if U is None:
            initial_U = numpy.asarray(
                numpy_rng.uniform(
                    low=-4 * numpy.sqrt(6. / (n_hidden + n_label)),
                    high=4 * numpy.sqrt(6. / (n_hidden + n_label)),
                    size=(n_label, n_hidden)
                ),
                dtype=theano.config.floatX
            )
            # theano shared variables for weights and biases
            U = theano.shared(value=initial_U, name='U', borrow=True)

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

        if lbias is None:
            # create shared variable for visible units bias
            lbias = theano.shared(
                value=numpy.zeros(
                    n_label,
                    dtype=theano.config.floatX
                ),
                name='lbias',
                borrow=True
            )

        # initialize input layer for standalone RBM or layer0 of DBN
        self.input = input
        if not input:
            self.input = T.matrix('input')
        self.label = label
        if not label:
            self.label = T.matrix('label')

        self.W = W
        self.U = U
        self.hbias = hbias
        self.vbias = vbias
        self.lbias = lbias

        self.theano_rng = theano_rng
        # **** WARNING: It is not a good idea to put things in this list
        # other than shared variables created in this function.
        self.params = [self.W, self.U, self.hbias, self.vbias, self.lbias]
        # end-snippet-1

    def free_energy(self, v_sample):
        ''' Function to compute the free energy '''
        label_term = T.dot(l_sample, self.lbias)
        visible_term = 0#T.dot(v_sample,self.vbias)
        hidden_preterm = self.hbias + T.dot(v_sample, self.W)+ T.dot(l_sample, self.U)
        hidden_term = T.sum(T.log(1 + T.exp(hidden_preterm)), axis=1)
        return -label_term -visible_term - hidden_term

    def propup(self, vis, label):
        '''This function propagates the visible units activation and the label units upwards to
        the hidden units'''
        pre_sigmoid_activation = self.hbias + T.dot(vis, self.W) + T.dot(label, self.U)
        return [pre_sigmoid_activation, T.nnet.sigmoid(pre_sigmoid_activation)]

    def sample_h_given_v(self, v0_sample, l0_sample):
        ''' This function infers state of hidden units given visible units '''
        pre_sigmoid_h1, h1_mean = self.propup(v0_sample, l0_sample)
        h1_sample = self.theano_rng.binomial(size=h1_mean.shape,
                                             n=1, p=h1_mean,
                                             dtype=theano.config.floatX) #sample with binomial law
        return [pre_sigmoid_h1, h1_mean, h1_sample]


    def propdown_vis(self, hid):
        '''This function propagates the hidden units activation downwards to the visible units'''
        pre_sigmoid_activation = self.vbias + T.dot(hid, self.W.T)
        return [pre_sigmoid_activation, T.nnet.sigmoid(pre_sigmoid_activation)]

    def propdown_label(self, hid):
        pre_softmax_activation = T.dot(hid, self.U.T) + self.lbias
        return [pre_softmax_activation, T.nnet.softmax(pre_softmax_activation)] #cf softmax theano : http://deeplearning.net/tutorial/logreg.html

    def sample_v_given_h(self, h0_sample):
        ''' This function infers state of visible units given hidden units '''
        pre_sigmoid_v1, v1_mean = self.propdown_vis(h0_sample)
        pre_softmax_l1, l1_mean =  self.propdown_label(h0_sample)

        v1_sample = self.theano_rng.binomial(size=v1_mean.shape,
                                             n=1, p=v1_mean,
                                             dtype=theano.config.floatX)
        l1_sample = self.theano_rng.binomial(size=l1_mean.shape,
                                             n=1, p=l1_mean,
                                             dtype=theano.config.floatX)
        return [pre_sigmoid_v1, v1_mean, v1_sample, pre_softmax_l1, l1_mean, l1_sample]

    def gibbs_hvh(self, h0_sample):
        ''' This function implements one step of Gibbs sampling, starting from the hidden state'''
        pre_sigmoid_v1, v1_mean, v1_sample, pre_softmax_l1, l1_mean, l1_sample = self.sample_v_given_h(h0_sample)
        pre_sigmoid_h1, h1_mean, h1_sample = self.sample_h_given_v(v1_sample, l1_sample)
        return [pre_sigmoid_v1, v1_mean, v1_sample, pre_softmax_l1, l1_mean, l1_sample,
                pre_sigmoid_h1, h1_mean, h1_sample]

    def gibbs_vhv(self, v0_sample, l0_sample):
        ''' This function implements one step of Gibbs sampling,
            starting from the visible and label state'''
        pre_sigmoid_h1, h1_mean, h1_sample = self.sample_h_given_v(v0_sample, l0_sample)
        pre_sigmoid_v1, v1_mean, v1_sample, pre_softmax_l1, l1_mean, l1_sample = self.sample_v_given_h(h1_sample)
        return [pre_sigmoid_h1, h1_mean, h1_sample,
                pre_sigmoid_v1, v1_mean, v1_sample, pre_softmax_l1, l1_mean, l1_sample]

    def p_y_given_x(self, v0_sample, label_expected):
        pre_p_y_given_x = -rbm.free_energy(v0_sample, label_expected)
        return T.nnet.softmax(pre_p_y_given_x)

    # start-snippet-2
    def get_cost_updates(self, lr=0.1, k=1):
        """This functions implements one step of CD-k
        :param lr: learning rate used to train the RBM
        :param k: number of Gibbs steps to do in CD-k
        Returns a proxy for the cost and the updates dictionary. The
        dictionary contains the update rules for weights and biases but
        also an update of the shared variable used to store the persistent
        chain, if one is used.

        """

        # compute positive phase
        pre_sigmoid_ph, ph_mean, ph_sample = self.sample_h_given_v(self.input, self.label)
        # decide how to initialize persistent chain:
        # for CD, we use the newly generate hidden sample
        chain_start = ph_sample

        # end-snippet-2
        # perform actual negative phase
        # in order to implement CD-k we need to scan over the
        # function that implements one gibbs step k times.
        # Read Theano tutorial on scan for more information :
        # http://deeplearning.net/software/theano/library/scan.html
        # the scan will return the entire Gibbs chain
        ([pre_sigmoid_nvs, nv_means, nv_samples, pre_softmax_lvs, nl_means, nl_samples,
          pre_sigmoid_nhs, nh_means, nh_samples], updates) = theano.scan(
            self.gibbs_hvh,
            # the None are place holders, saying that
            # chain_start is the initial state corresponding to the
            # 6th output
            outputs_info=[None, None, None, None, None, None, None, None, chain_start],
            n_steps=k
        )

        # start-snippet-3
        # determine gradients on RBM parameters
        # note that we only need the sample at the end of the chain
        chain_end = nv_samples[-1]
        label_end = nl_samples[-1]

        cost = T.mean(self.free_energy(self.input, self.label)) - T.mean(self.free_energy(chain_end, label_end))

        # We must not compute the gradient through the gibbs sampling
        #gparams = T.grad(cost, self.params, consider_constant=[chain_end])
        gparams = T.grad(cost, self.params, consider_constant=[chain_end, label_end]) #TODO really sure?

        # end-snippet-3 start-snippet-4
        # constructs the update dictionary
        for gparam, param in zip(gparams, self.params):
            # make sure that the learning rate is of the right dtype
            updates[param] = param - gparam * T.cast(lr,dtype=theano.config.floatX)
            monitoring_cost = self.get_reconstruction_cost(updates, pre_sigmoid_nvs[-1], pre_softmax_lvs[-1])

        return monitoring_cost, updates
        # end-snippet-4

    def get_reconstruction_cost(self, updates, pre_sigmoid_nv, pre_softmax_lv):
        """Approximation to the reconstruction error"""
        #cross_entropy = T.mean(T.sum(self.input * T.log(T.nnet.sigmoid(pre_sigmoid_nv)) +(1 - self.input) * T.log(1 - T.nnet.sigmoid(pre_sigmoid_nv)),axis=1))
        cross_entropy = T.mean(T.sum(self.label * T.log(T.nnet.softmax(pre_softmax_lv)) + (1 - self.label) * T.log(1 - T.nnet.softmax(pre_softmax_lv)), axis=1)) + \
                        T.mean(T.sum(self.input * T.log(T.nnet.sigmoid(pre_sigmoid_nv)) + (1 - self.input) * T.log(1 - T.nnet.sigmoid(pre_sigmoid_nv)),axis=1))
        return cross_entropy


def train_drbm(learning_rate=0.1, training_epochs=15,
             dataset='mnist.pkl.gz', batch_size=20,
             n_chains=20, n_samples=10, output_folder='rbm_plots',
             n_hidden=500, n_label=10):
    """
    Demonstrate how to train and afterwards sample from it using Theano.
    This is demonstrated on MNIST.
    :param learning_rate: learning rate used for training the RBM
    :param training_epochs: number of epochs used for training
    :param dataset: path the the pickled dataset
    :param batch_size: size of a batch used to train the RBM
    :param n_chains: number of parallel Gibbs chains to be used for sampling
    :param n_samples: number of samples to plot for each chain
    """
    rng = numpy.random.RandomState(123)
    theano_rng = RandomStreams(rng.randint(2 ** 30))

    datasets = load_data(dataset)
    train_set_x, train_set_y = datasets[0]
    test_set_x, test_set_y = datasets[2]

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size

    # allocate symbolic variables for the data
    index = T.lscalar()    # index to a [mini]batch
    x = T.matrix('x')  # the data is presented as rasterized images
    y = T.ivector('y')  # the label

    # construct the RBM class
    drbm = DRBM(input=x, label=y, n_visible=28 * 28, n_hidden=n_hidden, n_label=n_label, numpy_rng=rng, theano_rng=theano_rng)

    # get the cost and the gradient corresponding to one step of CD-15
    cost, updates = rbm.get_cost_updates(lr=learning_rate, k=15) #TODO CD-1 ?

    #################################
    #     Training the RBM          #
    #################################
    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)
    os.chdir(output_folder)

    # start-snippet-5
    # it is ok for a theano function to have no output
    # the purpose of train_rbm is solely to update the RBM parameters
    train_rbm = theano.function(
        [index],
        cost,
        updates=updates,
        givens={ x: train_set_x[index * batch_size: (index + 1) * batch_size],
                 y: train_set_y[index * batch_size: (index + 1) * batch_size]
        },
        name='train_rbm'
    )

    plotting_time = 0.
    start_time = timeit.default_timer()

    # go through training epochs
    for epoch in xrange(training_epochs):

        # go through the training set
        mean_cost = []
        for batch_index in xrange(n_train_batches):
            mean_cost += [train_rbm(batch_index)]

        print 'Training epoch %d, cost is ' % epoch, numpy.mean(mean_cost)

        # Plot filters after each training epoch
        plotting_start = timeit.default_timer()
        # Construct image from the weight matrix
        image = Image.fromarray(
            tile_raster_images(
                X=rbm.W.get_value(borrow=True).T,
                img_shape=(28, 28),
                tile_shape=(10, 10),
                tile_spacing=(1, 1)
            )
        )
        image.save('filters_at_epoch_%i.png' % epoch)
        plotting_stop = timeit.default_timer()
        plotting_time += (plotting_stop - plotting_start)

    end_time = timeit.default_timer()
    pretraining_time = (end_time - start_time) - plotting_time

    print ('Training took %f minutes' % (pretraining_time / 60.))
    return drbm

if __name__ == '__main__':
    #train model
    drbm = train_drbm()
    with open('best_model_drbm.pkl', 'w') as f:
        cPickle.dump(drbm, f)
    #test model
