import theano
import theano.tensor as TT
import numpy


my0 = TT.constant(numpy.array(0.0,dtype=theano.config.floatX))
my1 = TT.constant(numpy.array(1.0,dtype=theano.config.floatX))


def sgd(parameters,cost=None,gradients=None,
        updates=None,lr=2e-2, consider_constant = [],
        momentum = None, **kwargs):

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
        if momentum != None:
            mparam = theano.shared(param.get_value()*0.)
            updates[param] = param - scale * lr * mparam
            updates[mparam] = mparam*momentum + (1.-momentum)*grad
        else:
            updates[param] =  param - scale*lr * grad

    return updates, None

