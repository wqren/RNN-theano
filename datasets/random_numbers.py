"""
 Ops repesenting different datasets for the RNN.
"""

__authors__ = "Razvan Pascanu"
__copyright__ = "(c) 2011, Universite de Montreal"
__contact__ = "Razvan Pascanu <r.pascanu@gmail>"


import numpy
import theano
import theano.tensor as TT

class random_numbers(theano.Op):
    """
    TODO
    """
    def __init__( self
                 , n_outs
                 , style
                 , base_length
                 , random_length
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
        self.style         = style
        self.base_length   = base_length
        self.random_length = random_length
        self.max_val       = max_val
        self.min_val       = min_val
        self.batches       = batches
        self.batch_size    = batch_size
        self.noise         = noise
        self.data_u        = None
        self.data_t        = None
        self.index         = -1
        self.dtype         = dtype
        if style == 'continous':
            self.n_ins = 1
        elif style == 'random_pos':
            self.n_ins = 2
        else:
            raise NotImplemented

    def make_node(self, idx):
        input_seq = TT.tensor3( name = 'input', dtype = self.dtype)
        target    = TT.matrix( name = 'target', dtype = self.dtype)
        return theano.Apply(self, [idx], [input_seq, target])

    def __str__(self):
        return "%s[%s]" % (self.__class__.__name__, self.style)

    def grad(self, inputs, g_outputs):
        return [ None for i in inputs ]

    def refresh(self):
        if self.style == 'continous':
           shape=(self.batches
                  , self.base_length
                  , self.n_ins
                  , self.batch_size)

           self.data_u = numpy.zeros(shape)
           self.data_u[:,:self.n_outs] = \
                   self.rng.uniform(
                       low = self.min_val
                       , high = self.max_val
                       , size = (self.batches
                                 , self.n_outs
                                 , self.n_ins
                                 , self.batch_size))
           self.data_t = self.data_u[:,:self.n_outs,0,:]
           #if len(self.data_t.shape) == 2:
           #    self.data_t = self.data_t.reshape((1,)+self.data_t.shape)
           if self.noise > 0 :
                self.data_u = self.data_u + self.rng.uniform(
                    low = -self.noise
                    , high = self.noise
                    , size = self.data_u.shape)
           self.index += 1
        elif self.style == 'random_pos':
            shape = (self.batches
                     , self.base_length
                     , self.n_ins
                     , self.batch_size)

            self.data_u = numpy.zeros(shape)
            self.data_u[:,:,0,:] = self.rng.uniform( low = self.min_val
                                           , high = self.max_val
                                           , size = (
                                               self.batches
                                               , self.base_length
                                               , self.batch_size))

            self.data_t = numpy.zeros((self.batches
                                       , self.n_outs
                                       , self.batch_size))

            for dx in xrange(self.batches):
                for dy in xrange(self.batch_size):
                    p = self.rng.permutation(self.random_length)[:self.n_outs]
                    p.sort(axis=0)
                    self.data_t[dx,:,dy] = self.data_u[dx,:,0,dy][p]
                    self.data_u[dx,:,1,dy][p] = 1.
            if self.noise > 0 :
                self.data_u = self.data_u + self.rng.uniform(
                    low = -self.noise
                    , high = self.noise
                    , size = self.data_u.shape)
            self.index += 1
        else:
            raise NotImplemented

        self.data_u = numpy.asarray(self.data_u, dtype = self.dtype)
        self.data_t = numpy.asarray(self.data_t, dtype = self.dtype)


    def get_whole_tensors(self):
        if self.data_u is None or self.data_t is None:
            self.refresh()
        shared_u = theano.shared(self.data_u)
        shared_t = theano.shared(self.data_t)
        return shared_u, shared_t

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

    data_gen = random_numbers(
                    n_outs          = 2
                    , style         = 'continous'
                    , base_length   =  10
                    , random_length = 0
                    , max_val       = 1.
                    , min_val       = 0.
                    , batches       = 2
                    , batch_size    = 2
                    , noise         = 1e-5
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

    print '*************************************'
    print '*********** TASK II .................'
    data_gen = random_numbers(
                    n_outs          = 20
                    , style         = 'random_pos'
                    , base_length   =  100
                    , random_length = 50
                    , max_val       = 1.
                    , min_val       = 0.
                    , batches       = 2
                    , batch_size    = 1000
                    , noise         = 0.00
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
