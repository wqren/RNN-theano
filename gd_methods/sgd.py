import theano
import theano.tensor as TT
import numpy


my0 = TT.constant(numpy.array(0.0,dtype=theano.config.floatX))
my1 = TT.constant(numpy.array(1.0,dtype=theano.config.floatX))


def sgd(parameters,cost=None,gradients=None, stop=None,
                        updates=None,lr=2e-2, consider_constant = [],
                        **kwargs):

    if not isinstance(parameters,list):
        parameters = [parameters]
    if gradients == None:
        grads = TT.grad(cost,parameters,consider_constant = consider_constant)

    if updates==None:
        updates = {}
    for param,grad in zip(parameters,grads):
        scale = my1
        if 'scale' in kwargs:
            print 'scaling the lr'
            scale = kwargs['scale']
        updates[param] =  param - scale*lr * grad

    return updates, None

