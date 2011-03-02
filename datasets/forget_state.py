"""
 Ops repesenting different datasets for the RNN.
"""

__authors__ = "Razvan Pascanu"
__copyright__ = "(c) 2011, Universite de Montreal"
__contact__ = "Razvan Pascanu <r.pascanu@gmail>"



import numpy
import theano
import theano.tensor as TT

class forget_state(theano.Op):
    """
    TODO
    """
    def __init__( self
                 , n_states
                 , T
                 , switches
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
        assert n_states % 2 == 0
        self.n_states      = n_states
        self.T             = T
        self.switches      = switches
        self.batches       = batches
        self.batch_size    = batch_size
        self.noise         = noise
        self.data_u        = None
        self.data_t        = None
        self.index         = -1
        self.dtype         = dtype
        self.n_ins         = 1
        half_states        = n_states/2
        self.states        = [ -k*(1./(half_states+2)) for k in
                              xrange(1,half_states+1) ]
        self.states       += [ k*(1./(half_states+2)) for k in
                              xrange(1,half_states+1) ]


    def make_node(self, idx):
        input_seq = TT.tensor3( name = 'input', dtype = self.dtype)
        target    = TT.tensor3( name = 'target', dtype = self.dtype)
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
        shape = (self.batches
                 , self.T
                 , self.n_ins
                 , self.batch_size)

        self.data_u = numpy.zeros(shape)
        self.data_t = numpy.zeros(shape)

        for dx in xrange(self.batches):
            for dy in xrange(self.batch_size):
                p = self.rng.permutation(self.T)[:self.switches]
                p.sort(axis=0)

                for idx in xrange(self.switches) :
                    new_state = self.states[self.rng.randint(self.n_states)]
                    b = p[idx]
                    try:
                        e = p[idx+1]
                    except:
                        e = self.T
                    self.data_u[dx,b,0, dy] = new_state
                    self.data_t[dx,b:e,0,dy] = new_state

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

    def get_slice(idx):
        return self.data_u[idx], self.data_t[idx]

if __name__=='__main__':
    # Test of the op

    data_gen = forget_state(
                    n_states          = 4
                    , T             =  10
                    , switches       = 4
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
        print data_gen.states

