"""
 Initialization formulas for weight matrices.
"""

__authors__ = "Razvan Pascanu"
__copyright__ = "(c) 2011, Universite de Montreal"
__contact__ = "Razvan Pascanu <r.pascanu@gmail>"


import numpy
import theano
import utils


def init(n,m, name,style, properties = None, rng = None, mask = None):
    n = int(n)
    m = int(m)
    if rng == None:
        rng = numpy.random.RandomState(rng.random.randint(1e6))
    if style == 'orthogonal':
        assert n == m
        values = rng.uniform(size = (n,n))
        u,_,_  = numpy.linalg.svd(values)
        scale  = 1.
        if 'scale' in properties:
            scale = properties['scale']
        u      = numpy.asarray(u*scale, dtype = theano.config.floatX)
        return utils.shared_shape( u, name = name)
    elif style == 'random':
        scale = 1./n
        if 'scale' in properties:
            scale = properties['scale']
        values = rng.uniform(low = -scale, high = scale, size=(n*m,))
        sparsity = 0.
        if 'sparsity' in properties:
            sparsity = properties['sparsity']
        positions = rng.permutation(n*m)
        values[positions[:int(n*m*sparsity)]] = 0.
        values = values.reshape((n,m))
        values = numpy.asarray(values, theano.config.floatX)
        return utils.shared_shape(values, name)
    elif style == 'esn':
        assert n == m
        trials    = 0
        success = False
        while not success:
            try:
                values = rng.uniform(low = -1, high = 1, size=(n*m,))
                sparsity = 0.9
                if 'sparsity' in properties:
                    sparsity = properties['sparsity']
                positions = rng.permutation(n*m)
                limit = int(n*n*sparsity)
                if n < 30 :
                    limit = n*n-n
                values[positions[:limit]] = 0.
                values = values.reshape((n,n))
                maxval = numpy.max(numpy.abs(numpy.linalg.eigvals(values)))
                scale = 0.5
                if 'scale' in properties:
                    scale = properties['scale']
                values = values * scale / maxval
                success = True
            except:
                print 'ESN weights generation, trail', trails
                trails += 1
                if trails > 20:
                    raise ValueError('Could not generate ESN weights')
        if mask:
            values = values * mask
        values =  numpy.asarray(values, dtype = theano.config.floatX)
        return utils.shared_shape(values, name = name)



