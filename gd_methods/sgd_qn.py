import theano
import theano.tensor as TT
import numpy


my0 = TT.constant(numpy.array(0.0,dtype=theano.config.floatX))
my1 = TT.constant(numpy.array(1.0,dtype=theano.config.floatX))

def sgd_qn(parameters,cost=None,gradients=None, stop=None,
                        updates=None,mylambda=1e-4,t0=1e5,skip=16,
                        consider_constant = None,lazy = False,
                        **kwargs):

    # lazy condition. Needs mode = LazyLinker() when building the function
    if lazy:
        from theano.lazycond import cond as ifelse
    else:
        from theano.tensor import switch as ifelse

    if not isinstance(parameters,list):
        parameters = [parameters]

    # We need a copy of the parameters:
    new_parameters = []
    replace_dict   = {}
    for p in parameters:
        np = theano.shared( p.value.copy() )
        new_parameters.append(np)
        replace_dict[p] = np

    # cloning the graph, equiv_dict[replaced_inputs[i]] contains the input in the
    # cloned graph that corresponds to the input in the original graph.
    # likewise, equiv_dict[cost] is the variable in the cloned graph,
    # corresponding to the cost
    new_cost = theano.clone(cost, replace = replace_dict)

    if gradients == None:
        # for RBM-like models, parts of the graph need to be in
        # "consider_constant"
        if consider_constant != None:
            if not isinstance(consider_constant,list):
                consider_constant = [consider_constant]
            grads = TT.grad(cost,parameters,
                            consider_constant=consider_constant)
            new_param_grads = TT.grad(new_cost,new_parameters,
                                      consider_constant = consider_constant)
        else:
            grads = TT.grad(cost,parameters)
            new_param_grads = TT.grad(new_cost,new_parameters)

    # For graph readability & debugging
    for p,np,gp,gnp in zip(parameters,new_parameters,grads,new_param_grads):
        np.name = p.name + '_o'
        if gp.name != None:
            gnp.name = gp.name + '_o'
        else:
            gp.name = 'g_' + p.name
            gnp.name = 'g_' + p.name + '_o'

    grad_diff = [g - ng for g,ng in zip(grads,new_param_grads)]
    param_diff = [p - np for p,np in zip(parameters,new_parameters)]

    #the_ratios = [TT.clip(gd/pd,mylambda,100.*mylambda) for gd,pd in
    #                                     zip(grad_diff,param_diff)]
    from utils import true_div_special # if 0/0, replace with mylambda
    div = TT.Elemwise(true_div_special)
    the_ratios = [TT.clip(div(gd,pd,mylambda),mylambda,
                 numpy.array(100.
                             ,dtype=theano.config.floatX )*mylambda)
                  for gd,pd in zip(grad_diff,param_diff)]

    # allocate a B (the "learning rates") for each param
    b_list = []
    for param in parameters:
        b_init = numpy.ones_like(param.value) / (mylambda * t0)
        b = theano.shared(value = b_init, name = 'b_'+param.name)
        b_list.append(b)

    updateB = theano.shared(numpy.array(0.0,dtype=theano.config.floatX)
                            , name='updateB')
    count = theano.shared(numpy.array(skip,dtype=theano.config.floatX)
                          , name='count')

    # build the update dictionary
    if updates == None:
        updates = {}


    myskip = theano.shared(numpy.array(skip,dtype=theano.config.floatX))

    # updates for counters
    updates[updateB] = ifelse(TT.eq(count,my1),my1,my0)
    updates[count] = ifelse(TT.le(count,my0),myskip,count - my1)

    for b,ratio in zip(b_list,the_ratios):
        updates[b] = ifelse(TT.eq(updateB,my1),
                                    b / (my1 + skip * b * ratio),
                                    b)

    for new_param,param in zip(new_parameters,parameters):
        updates[new_param] = ifelse(TT.le(count,my0),
                                    param,
                                    new_param)

    for param,b,grad,new_grad in zip(parameters,b_list,grads,new_param_grads):
        scale = my1
        if 'scale' in kwargs:
            print 'scaling the lr'
            scale = kwargs['scale']
        updates[param] = ifelse(TT.le(count,my0),
                                      param - scale*b * new_grad,
                                      param - scale*b * grad)

    extras = [count, updateB, b_list
              , parameters,grads,new_parameters,new_param_grads]

    return updates,extras

