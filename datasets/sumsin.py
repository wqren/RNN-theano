"""
 Ops repesenting different datasets for the RNN.
"""

__authors__ = "Razvan Pascanu"
__copyright__ = "(c) 2011, Universite de Montreal"
__contact__ = "Razvan Pascanu <r.pascanu@gmail>"



import numpy
import theano
import theano.tensor as TT

class sumsin(theano.Op):
    """
    TODO
    """
    def __init__( self
                 , T
                 , steps
                 , batches
                 , batch_size
                 , noise = 0.
                 , rng = None
                 , dtype = theano.config.floatX
                ):
        if rng is None:
            rng = numpy.random.RandomState(numpy.random.randint(1e6))
        self.rng           = rng
        self.n_outs        = 1
        self.T             = T
        self.steps         = steps
        self.max_step      = numpy.max(self.steps)
        self.batches       = batches
        self.batch_size    = batch_size
        self.noise         = noise
        self.data_u        = None
        self.data_t        = None
        self.index         = -1
        self.dtype         = dtype
        self.n_ins         = 1
        self.n_outs        = len(self.steps)
        ct = numpy.arange(self.T+self.max_step)
        sinwave = numpy.sin(ct*0.2) + numpy.sin(0.311*ct) + \
                        numpy.sin(0.42*ct)
        self.data_u = numpy.zeros((self.T, 1))
        self.data_u[:,0] = sinwave[:self.T]
        self.data_t = numpy.zeros((self.n_outs, self.T))
        for k in xrange(self.n_outs):
            self.data_t[k,:] = sinwave[self.steps[k]: self.T + self.steps[k]]
        self.data_u = numpy.asarray(self.data_u, dtype = dtype)
        self.data_t = numpy.asarray(self.data_t, dtype = dtype)

    def make_node(self, idx):
        input_seq = TT.matrix( name = 'input', dtype = self.dtype)
        input_seq.tag.shape = [ self.T, self.n_ins ]
        target    = TT.matrix( name = 'target', dtype = self.dtype)
        target.tag.shape = [ self.n_outs, self.T ]
        return theano.Apply(self, [idx], [input_seq, target])

    def __str__(self):
        return "%s" % self.__class__.__name__

    def clean(self):
        # clean memory
        del self.data_u
        del self.data_t
        self.data_u = None
        self.data_t = None

    def grad(self, inputs, g_outputs):
        return [ None for i in inputs ]

    def refresh(self):
        pass


    def get_whole_tensors(self):
        if self.data_u is None or self.data_t is None:
            self.refresh()
        shared_u = theano.shared(self.data_u)
        shared_t = theano.shared(self.data_t)
        return shared_u, shared_t


    def infer_shape(self, node, input_shapes):
        return [(self.T, self.n_ins, self.batch_size),
                (self.T, self.n_outs, self.batch_size)]

    def perform (self, node, inputs, (sequence, target)):
        if self.noise > 0 :
            sequence[0] = self.data_u + numpy.random.uniform(
                size=self.data_u.shape, low = -self.noise,
                                 high = self.noise)
        else:
            sequence[0] = self.data_u
        target[0] = self.data_t

    def get_slice(self):
        if self.noise > 0:
            seq = self.data_u + numpy.random.uniform(
                size=self.data_u.shape, low = -self.noise,
                                 high = self.noise)
        else:
            seeq = self.data_u
        return self.data_u, self.data_t

if __name__=='__main__':
    # Test of the op

    data_gen = sumsin(
                     T             =  10
                    , steps       = [1]
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
