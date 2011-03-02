"""
 Ops repesenting different datasets for the RNN.
"""

__authors__ = "Razvan Pascanu"
__copyright__ = "(c) 2011, Universite de Montreal"
__contact__ = "Razvan Pascanu <r.pascanu@gmail>"



import numpy
import theano
import theano.tensor as TT

class spike_numbers(theano.Op):
    """
    TODO
    """
    def __init__( self
                 , n_outs
                 , T
                 , inrange
                 , max_val
                 , min_val
                 , batches
                 , batch_size
                 , noise = 0.
                 , rng = None
                 , dtype = theano.config.floatX
                ):
        if rng is None:
            rng = numpy.random.RandomState(numpy.random.randint(1e6))
        self.rng           = rng
        self.n_outs        = n_outs
        self.T             = T
        self.inrange       = inrange
        self.max_val       = max_val
        self.min_val       = min_val
        self.batches       = batches
        self.batch_size    = batch_size
        self.noise         = noise
        self.data_u        = None
        self.data_t        = None
        self.index         = -1
        self.dtype         = dtype
        self.n_ins = 2

    def make_node(self, idx):
        input_seq = TT.tensor3( name = 'input', dtype = self.dtype)
        target    = TT.matrix( name = 'target', dtype = self.dtype)
        return theano.Apply(self, [idx], [input_seq, target])

    def __str__(self):
        return "%s" % self.__class__.__name__

    def grad(self, inputs, g_outputs):
        return [ None for i in inputs ]

    def clean(self):
        # clean memory
        del self.data_u
        del self.data_t
        self.data_u = None
        self.data_t = None


    def refresh(self):
        shape = (self.batches
                 , self.T
                 , self.n_ins
                 , self.batch_size)

        self.data_u = numpy.zeros(shape)
        self.data_u[:,:,0,:] = self.rng.uniform( low = self.min_val
                                       , high = self.max_val
                                       , size = (
                                           self.batches
                                           , self.T
                                           , self.batch_size))

        self.data_t = numpy.zeros((self.batches
                                   , self.n_outs
                                   , self.batch_size))

        for dx in xrange(self.batches):
            for dy in xrange(self.batch_size):
                p = self.rng.permutation(self.inrange)[:self.n_outs]
                p.sort(axis=0)
                self.data_t[dx,:,dy] = self.data_u[dx,:,0,dy][p]
                self.data_u[dx,:,1,dy][p] = 1.
        if self.noise > 0 :
            self.data_u = self.data_u + self.rng.uniform(
                low = -self.noise
                , high = self.noise
                , size = self.data_u.shape)
        self.index += 1

        self.data_u = numpy.asarray(self.data_u, dtype = self.dtype)
        self.data_t = numpy.asarray(self.data_t, dtype = self.dtype)


    def get_whole_tensors(self):
        if self.data_u is None or self.data_t is None:
            self.refresh()
        shared_u = theano.shared(self.data_u)
        shared_t = theano.shared(self.data_t)
        return shared_u, shared_t


    def infer_shape(self, node, input_shapes):
        return [(self.T, self.n_ins, self.batch_size),
                (self.n_outs, self.batch_size)]

    def perform (self, node, inputs, (sequence, target)):
        if self.index < 0:
            self.refresh()
        idx = inputs[0] - self.index*self.batches
        if idx < 0:
            idx = idx % self.batches
        while idx >= self.batches:
            self.refresh()

        sequence[0] =  self.data_u[idx]
        target[0]   =  self.data_t[idx]

if __name__=='__main__':
    # Test of the op

    data_gen = spike_numbers(
                    n_outs          = 2
                    , T             =  10
                    , inrange       = 5
                    , max_val       = 1.
                    , min_val       = 0.
                    , batches       = 2
                    , batch_size    = 2
                    , noise         = 0
                    )
    idx = theano.tensor.scalar()
    s,t = data_gen(idx)
    f   = theano.function([idx], [s,t])
    for idx in xrange(2):
        print 'Entry #',idx,' :'
        seq,target = f(idx)
        print '--Sequence--'
        print seq
        print '--Target---'
        print target
        print

