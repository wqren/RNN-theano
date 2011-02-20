#!/usr/bin/env python
"""
 Template of a simple regularized/non-regularized RNN network on a task.
 13 Feb 2011
"""

__authors__ = "Razvan Pascanu"
__copyright__ = "(c) 2011, Universite de Montreal"
__contact__ = "Razvan Pascanu <r.pascanu@gmail>"

import ConfigParser, cPickle, gzip, numpy, time, optparse, os

import theano
import theano.tensor as TT
import theano_linalg
from theano.gof.link import LazyLinker
from theano import function, Mode


from RNN_theano.utils.init_mat import init
from RNN_theano.gd_methods.sgd import sgd
from RNN_theano.gd_methods.sgd_qn import sgd_qn
from RNN_theano.utils.utils import parse_input_arguments, floatX
from RNN_theano.datasets.random_numbers import random_numbers

# Using the correct dtype for constatsns
my0 = numpy.asarray(0., dtype = theano.config.floatX)
my1 = numpy.asarray(1., dtype = theano.config.floatX)






def jobman(_options, channel = None):

    ################### PARSE INPUT ARGUMENTS #######################
    o = parse_input_arguments(_options, 'templaterc.ini')
    ####################### DEFINE THE TASK #########################

    rng = numpy.random.RandomState(o['seed'])
    train_set = random_numbers(
                    n_outs          = o['n_outs']
                    , style         = o['style']
                    , base_length   = o['task_base_length']
                    , random_length = o['task_random_length']
                    , max_val       = o['task_max_val']
                    , min_val       = o['task_min_val']
                    , batches       = o['task_train_batches']
                    , batch_size    = o['task_train_batchsize']
                    , noise         = o['task_noise']
                    , rng           = rng
                    )

    valid_set = random_numbers(
                    n_outs           = o['n_outs']
                    , style          = o['style']
                    , base_length    = o['task_base_length']
                    , random_length  = o['task_random_length']
                    , max_val        = o['task_max_val']
                    , min_val        = o['task_min_val']
                    , batches        = o['task_valid_batches']
                    , batch_size     = o['task_valid_batchsize']
                    , rng            = rng
                    )

    test_set = random_numbers(
                    n_outs          = o['n_outs']
                    , style         = o['style']
                    , base_length   = o['task_base_length']
                    , random_length = o['task_random_length']
                    , max_val       = o['task_max_val']
                    , min_val       = o['task_min_val']
                    , batches       = o['task_test_batches']
                    , batch_size    = o['task_test_batchsize']
                    , rng           = rng
                    )

    wout_set = random_numbers(
                    n_outs          = o['n_outs']
                    , style         = o['style']
                    , base_length   = o['task_base_length']
                    , random_length = o['task_random_length']
                    , max_val       = o['task_max_val']
                    , min_val       = o['task_min_val']
                    , batches       = o['task_wout_batches']
                    , batch_size    = o['task_wout_batchsize']
                    , noise         = o['task_wout_noise']
                    , rng           = rng
                    )

    ###################### DEFINE THE MODEL #########################

    def recurrent_fn( u_t, h_tm1, W_hh, W_ux, W_hy,b) :
        x_t = TT.dot(W_ux, u_t)
        h_t = TT.tanh( TT.dot(W_hh, h_tm1) + x_t + b)
        y_t = TT.dot(W_hy, h_t)
        return h_t, y_t

    u  = TT.tensor3('u')
    if o['error_over_all']:
        t = TT.tensor3('t')
    else:
        t  = TT.matrix('t')
    h0 = TT.matrix('h0')
    b  = theano.shared( floatX(numpy.zeros((o['nhid']))), name='b')


    W_hh = init( o['nhid']
                , o['nhid']
                , 'W_hh'
                , o['Whh_style']
                , o['Whh_properties']
                , rng)

    W_ux = init( o['nhid']
                , train_set.n_ins
                , 'W_ux'
                , o['Wux_style']
                , o['Wux_properties']
                , rng)

    W_hy = init( o['n_outs']
                , o['nhid']
                , 'W_hy'
                , o['Why_style']
                , o['Why_properties']
                , rng)
    [h,y], _ = theano.scan(
        recurrent_fn
        , sequences = u
        , outputs_info = [h0, None]
        , non_sequences = [W_hh, W_ux, W_hy, TT.shape_padright(b)]
        , name = 'recurrent_fn'
        )

    init_h =h.owner.inputs[0].owner.inputs[2]

    if o['error_over_all']:
        out_err = TT.mean(TT.mean((y-t)**2, axis = 0), axis=1)
        err     = out_err.mean()
    else:
        out_err = ((y[-1] -t)**2).mean(axis=1)
        err     = out_err.mean()
    # Regularization term
    if o['reg_projection'] == 'h[-1]':
        cost = h[-1].sum()
    elif o['reg_projection'] == 'err':
        cost = err
    elif o['reg_projection'] == 'random':
        trng = TT.shared_randomstreams.RandomStreams(rng.randint(1e6))
        proj = trng.uniform(size = h[-1].shape)
        cost = TT.sum(proj*h[-1])

    z,gh = TT.grad(cost, [init_h, h])
    z = z[:-1] -gh
    z2 = TT.sum(z**2, axis = 2)
    v1 = z2[:-1]
    v2 = z2[1:]
    ratios = TT.switch(TT.ge(v2,1e-7), TT.sqrt(v1/v2), my1)
    norm_0 = TT.ones_like(ratios[0])
    norm_t, _ = theano.scan(lambda x,y: x*y
                            , sequences = ratios
                            , outputs_info = norm_0
                            , name = 'jacobian_products')
    norm_term = TT.sum(TT.mean(norm_t, axis=1))
    if o['reg_cost'] == 'product':
        r = TT.mean( abs(TT.log(norm_t)), axis=1).sum()
    elif o['reg_cost'] == 'each':
        r = TT.mean( abs(TT.log(ratios)), axis=1).sum()
    elif o['reg_cost'] == 'product2':
        ratios2 = TT.switch(TT.ge(z2[-1],1e-7), TT.sqrt(z2/z2[-1]), my1)
        r = TT.mean( abs(TT.log(ratios2)), axis=1).sum()

    gu = TT.grad(y[-1].sum(), u)

    if o['opt_alg'] == 'sgd':
        get_updates = lambda p,e, up : ( sgd(p
                                           , e
                                           , lr      = floatX(o['lr'])
                                           , scale   =  my1/norm_term
                                           , updates = up)[0]
                                        , [[],[],[TT.constant(0) for x in p]] )
    elif o['opt_alg'] == 'sgd_qn':
        get_updates = lambda p,e, up : sgd_qn(p
                                              , e
                                              , mylambda = floatX(o['mylambda'])
                                              , t0 = floatX(o['t0'])
                                              , skip = floatX(o['skip'])
                                              , scale = my1/norm_term
                                              , lazy = o['lazy']
                                              , updates = up)

    if o['win_reg']:
        updates,why_extra = get_updates([W_hy], err, {})
        cost = err + floatX(o['alpha'])*r
        updates,extras = get_updates([W_ux, W_hh,b], cost, updates)
        b_Why = why_extra[2][0]
        b_Wux = extras[2][0]
        b_Whh = extras[2][1]
        b_b   = extras[2][2]
    else:
        updates,extras1 = get_updates([W_hy, W_ux], err, {})
        cost = err + floatX(o['alpha'])*r
        updates,extras2 = get_updates([W_hh,b], cost, updates)
        b_Why = extras1[2][0]
        b_Wux = extras1[2][1]
        b_Whh = extras2[2][0]
        b_b   = extras2[2][1]

    if o['lazy']:
        mode = Mode(linker=LazyLinker(), optimizer='fast_run')
    else:
        mode = None

    nhid = o['nhid']
    train_batchsize = o['task_train_batchsize']
    valid_batchsize = o['task_valid_batchsize']
    test_batchsize = o['task_test_batchsize']
    wout_batchsize = o['task_wout_batchsize']

    train_h0 = theano.shared(floatX(numpy.zeros((nhid,train_batchsize))))
    valid_h0 = theano.shared(floatX(numpy.zeros((nhid,valid_batchsize))))
    test_h0  = theano.shared(floatX(numpy.zeros((nhid,test_batchsize))))
    wout_h0  = theano.shared(floatX(numpy.zeros((nhid,wout_batchsize))))
    idx = TT.iscalar('idx')
    train_u, train_t = train_set(idx)
    train = theano.function([idx], [out_err, r, norm_term]
                            , updates = updates
                            , mode = mode
                            , givens = {   u: train_u
                                        ,  t: train_t
                                        , h0: train_h0
                                       } )
    valid_u, valid_t = valid_set(idx)
    valid = theano.function([idx], [out_err, r, norm_term]
                            , mode = mode
                            , givens = {   u: valid_u
                                        ,  t: valid_t
                                        , h0: valid_h0
                                       } )

    test_u, test_t = test_set(idx)
    test = theano.function([idx], [out_err
                                   , r
                                   , norm_term
                                   , W_hh
                                   , W_ux
                                   , W_hy
                                   , b
                                   , z
                                   , y
                                   , h
                                   , u
                                   , gu
                                   , t
                                   , b_Whh
                                   , b_Wux
                                   , b_Why
                                   , b_b]
                            , mode = mode
                            , givens = {   u: test_u
                                        ,  t: test_t
                                        , h0: test_h0
                                       } )

    wout_u, wout_t = wout_set.get_whole_tensors()


    def wiener_hopf_fn( u_t, t_t, H_tm1, Y_tm1, W_hh, W_ux, b, h0):
        def recurrent_fn(u_t, h_tm1, W_hh, W_ux, b):
            x_t = TT.dot(W_ux, u_t)
            h_t = TT.tanh( TT.dot(W_hh, h_tm1) + x_t + b)
            return h_t
        h_t, _ = theano.scan(
            recurrent_fn
            , sequences = u_t
            , outputs_info = h0
            , non_sequences = [W_hh, W_ux, b ]
            , name = 'recurrent_fn'
        )
        H_t = H_tm1 + TT.dot(h_t[-1], h_t[-1].T)
        Y_t = Y_tm1 + TT.dot(h_t[-1], t_t.T)
        return H_t, Y_t

    H_0 = theano.shared(numpy.zeros((o['nhid'], o['nhid'])
                                   , dtype = theano.config.floatX)
                        , name='H0')
    Y_0 = theano.shared(numpy.zeros((o['nhid'], o['n_outs'])
                                    , dtype = theano.config.floatX)
                        , name='Y0')
    all_u = TT.tensor4('whole_u')
    all_t = TT.tensor3('whole_t')
    [H,Y], _ = theano.scan(
        wiener_hopf_fn
        , sequences = [all_u,all_t]
        , outputs_info = [H_0, Y_0]
        , non_sequences = [W_hh, W_ux, TT.shape_padright(b), h0]
        , name = 'wiener_hopf_fn'
        )
    length = TT.cast(all_u.shape[0]*all_u.shape[3]
                     , dtype = theano.config.floatX)
    H = H[-1]/length
    Y = Y[-1]/length
    H = H + floatX(o['wiener_lambda'])*TT.eye(o['nhid'])
    W_hy_solve = theano_linalg.solve(H, Y).T
    wout = theano.function([idx], []
                           , mode = mode
                           , updates = {W_hy: W_hy_solve }
                           , givens = {   all_u: wout_u
                                       ,  all_t: wout_t
                                       , h0: wout_h0
                                      } )

    '''
    theano.printing.pydotprint(train, 'train.png', high_contrast=True)
    for idx, o in enumerate(train.maker.env.outputs):
        if o.owner.op.__class__.__name__ == 'Cond':
            theano.printing.pydotprint_variables([o.owner.inputs[1]]
                                                  , 'lazy%d_left.png'%idx
                                                  , high_contrast= True)

            theano.printing.pydotprint_variables([o.owner.inputs[2]]
                                                  , 'lazy%d_right.png'%idx
                                                  , high_contrast= True)
    '''
    #################### DEFINE THE MAIN LOOP #######################


    data = {}
    fix_len = int(o['NN']/o['small_step'])
    avg_train_err  = numpy.zeros((o['small_step'],o['n_outs']))
    avg_train_reg  = numpy.zeros((o['small_step'],))
    avg_train_norm = numpy.zeros((o['small_step'],))
    avg_valid_err  = numpy.zeros((o['small_step'],o['n_outs']))
    avg_valid_reg  = numpy.zeros((o['small_step'],))
    avg_valid_norm = numpy.zeros((o['small_step'],))

    data['options'] = o
    data['train_err']  = -1*numpy.ones((fix_len,o['n_outs']))
    data['valid_err']  = -1*numpy.ones((fix_len,o['n_outs']))
    data['train_reg']  = -1*numpy.ones((fix_len,))
    data['valid_reg']  = -1*numpy.ones((fix_len,))
    data['train_norm'] = numpy.zeros((fix_len,))
    data['valid_norm'] = numpy.zeros((fix_len,))

    data['test_err']  = [None]*o['max_storage']
    data['test_reg']  = [None]*o['max_storage']
    data['test_norm'] = [None]*o['max_storage']
    data['y']         = [None]*o['max_storage']
    data['z']         = [None]*o['max_storage']
    data['t']         = [None]*o['max_storage']
    data['h']         = [None]*o['max_storage']
    data['u']         = [None]*o['max_storage']
    data['gu']        = [None]*o['max_storage']
    data['W_hh']      = [None]*o['max_storage']
    data['W_ux']      = [None]*o['max_storage']
    data['W_hy']      = [None]*o['max_storage']
    data['b']         = [None]*o['max_storage']
    data['b_ux']      = [None]*o['max_storage']
    data['b_hy']      = [None]*o['max_storage']
    data['b_hh']      = [None]*o['max_storage']
    data['b_b']       = [None]*o['max_storage']
    stop = False

    old_rval = numpy.inf
    patience = o['patience']
    n_train = o['task_train_batches']
    n_valid = o['task_valid_batches']
    n_test  = o['task_test_batches']
    n_test_runs = -1
    test_pos    = -1
    for idx in xrange(int(o['NN'])):
        jdx = idx%o['small_step']
        avg_train_err[jdx,:] = 0
        avg_train_reg[jdx]   = 0
        avg_train_norm[jdx]  = 0

        avg_valid_err[jdx,:] = 0
        avg_valid_reg[jdx]   = 0
        avg_valid_norm[jdx]  = 0
        print '*Re-generate training set '
        st = time.time()
        train_set.refresh()
        print '**Generation took', time.time() - st, 'secs'
        st = time.time()
        for k in xrange(o['task_train_batches']):
            rval = train(k)
            print '[',idx,'][',k,'/',n_train,'][train]', rval[0].mean(), \
                    rval[1], rval[2]
            avg_train_err[jdx,:]  += rval[0]
            avg_train_reg[jdx]  += rval[1]
            avg_train_norm[jdx] += rval[2]
        print '**Epoch took', time.time() - st, 'secs'
        avg_train_err[jdx]  /= n_train
        avg_train_reg[jdx]  /= n_train
        avg_train_norm[jdx] /= n_train
        st = time.time()

        if o['wout_pinv'] and (idx%o['test_step'] == 0):
            wout_set.refresh()
            print ( '* Re-computing W_hy using closed-form '
                   'regularized wiener hopf formula')
            st = time.time()
            wout(0)
            ed = time.time()
            print '** It took ', ed-st,'secs'
            print '** Average weight', abs(W_hy.value).mean()


        valid_set.refresh()
        st = time.time()
        for k in xrange(n_valid):
            rval = valid(k)
            print '[',idx,'][',k,'/',n_valid,'][valid]', rval[0].mean(), \
                    rval[1], rval[2]
            avg_valid_err[jdx]  += rval[0]
            avg_valid_reg[jdx]  += rval[1]
            avg_valid_norm[jdx] += rval[2]

        avg_valid_err[jdx]  /= n_valid
        avg_valid_reg[jdx]  /= n_valid
        avg_valid_norm[jdx] /= n_valid
        if idx > o['small_step'] and idx%o['small_step'] == 0:
            kdx = int(idx /o['small_step'])
            data['train_err'][kdx]  = avg_train_err.mean()
            data['valid_err'][kdx]  = avg_valid_err.mean()
            data['train_reg'][kdx]  = avg_train_reg.mean()
            data['valid_reg'][kdx]  = avg_valid_reg.mean()
            data['train_norm'][kdx] = avg_train_norm.mean()
            data['valid_norm'][kdx] = avg_valid_norm.mean()
            if channel :
                try:
                    _options['trainerr']    = data['train_err'][kdx].mean()
                    _options['maxtrainerr'] = data['train_err'][kdx].max()
                    _options['trainreg']    = data['train_reg'][kdx]
                    _options['trainnorm']   = data['train_norm'][kdx]
                    _options['validerr']    = data['valid_err'][kdx].mean()
                    _options['maxvaliderr'] = data['valid_err'][kdx].max()
                    _options['validreg']    = data['valid_reg'][kdx]
                    _options['validnorm']   = data['valid_norm'][kdx]
                    _options['steps']       = idx
                    _options['patience']    = patience
                    channel.save()
                except:
                    pass

        print '** ', avg_valid_err[jdx].mean(), ' < ', old_rval, ' ? '
        if avg_valid_err[jdx].mean() < old_rval :

            patience += o['patience_incr']
            if avg_valid_err[jdx].mean() < old_rval*0.99:
                n_test_runs += 1
                test_pos    += 1
                if test_pos >= o['max_storage']:
                    test_pos = test_pos - o['go_back']

                test_err  = []
                test_reg  = []
                test_norm = []
                test_y    = []
                test_z    = []
                test_t    = []
                test_h    = []
                test_u    = []
                test_gu   = []

                test_set.refresh()
                for k in xrange(n_test):
                    rval = test(k)
                    print '[',idx,'][',k,'/',n_test,'][test]',rval[0].mean()\
                        , rval[1], rval[2]
                    test_err   += [rval[0]]
                    test_reg   += [rval[1]]
                    test_norm  += [rval[2]]
                    test_z     += [rval[7]]
                    test_y     += [rval[8]]
                    test_h     += [rval[9]]
                    test_u     += [rval[10]]
                    test_gu    += [rval[11]]
                    test_t     += [rval[12]]
                data['y'][test_pos]         = [ test_y    ]
                data['z'][test_pos]         = [ test_z    ]
                data['t'][test_pos]         = [ test_t    ]
                data['h'][test_pos]         = [ test_h    ]
                data['u'][test_pos]         = [ test_u    ]
                data['gu'][test_pos]        = [ test_gu   ]
                data['test_err'][test_pos]  = [ test_err  ]
                data['test_reg'][test_pos]  = [ test_reg  ]
                data['test_norm'][test_pos] = [ test_norm ]
                data['W_hh'][test_pos]      = [ rval[3]   ]
                data['W_ux'][test_pos]      = [ rval[4]   ]
                data['W_hy'][test_pos]      = [ rval[5]   ]
                data['b'][test_pos]         = [ rval[6]   ]
                data['b_hh'][test_pos]      = [ rval[13]  ]
                data['b_ux'][test_pos]      = [ rval[14]  ]
                data['b_hy'][test_pos]      = [ rval[15]  ]
                data['b_b'][test_pos]       = [ rval[16]  ]

                cPickle.dump(data,
                    open(os.path.join(o['path'],'%s.pkl'%o['name'])
                         ,'wb'))

            old_rval = avg_valid_err[jdx].mean()
        if idx > patience:
                break



def process_options():
    parser = optparse.OptionParser()

    parser.add_option( '-f'
                     , "--config-file"
                     , dest    = 'configfile'
                     , help    = 'Config file that holds default options'
                     )

    parser.add_option( "--NN"
                     , dest    = 'NN'
                     , help    = ('length of outer loop (i.e. number of'
                                  ' epochs, where each epoch has new data'
                                  ' generated on the fly')
                     , type = 'int'
                     )


    parser.add_option( "--N"
                     , dest    = 'N'
                     , help    = ('number of batches in epoch')
                     , type = 'int'
                     )


    parser.add_option( "--small_step"
                     , dest    = 'small_step'
                     , help    = ('step size for computing error, and other'
                                  ' stats')
                     , type = 'int'
                     )

    parser.add_option( "--large_step"
                     , dest    = 'large_step'
                     , help    = ('step size for saving weigths')
                     , type = 'int'
                     )

    parser.add_option( "--n_hid"
                     , dest    = 'nhid'
                     , help    = ('number of hidden units')
                     , type = 'int'
                     )


    parser.add_option( "--B"
                     , dest    = 'B'
                     , help    = ('batchsize')
                     , type = 'int'
                     )


    parser.add_option( "--r_lambda"
                     , dest    = 'r_lambda'
                     , help    = ('Regularization term weight')
                     , type = 'float'
                     )


    parser.add_option( "--name"
                     , dest    = 'name'
                     , help    = ('Name of running job')
                     )

    parser.add_option( "--seed"
                     , dest    = 'seed'
                     , help    = ('random seed')
                     , type = 'int'
                     )

    parser.add_option("-l"
                      , "--lazy"
                      , dest="lazy"
                      , action="store_true"
                      , default=False
                      , help="use LazyLinker")

    parser.add_option("-o"
                      , "--opt_alg"
                      , dest="opt_alg"
                      , help="which sgd optimizer to use")

    return parser.parse_args()


if __name__=='__main__':
    (options,args) = process_options()
    jobman(options.__dict__)

