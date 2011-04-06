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
import configs
import copy

#from theano.gof.link import LazyLinker
from theano import function, Mode
from theano.gof import vm

from RNN_theano.utils.init_mat import init
from RNN_theano.gd_methods.sgd import sgd
from RNN_theano.gd_methods.sgd_qn import sgd_qn
from RNN_theano.utils.utils import parse_input_arguments, floatX,\
                                shared_shape
from RNN_theano.datasets.spike_numbers import spike_numbers
from RNN_theano.datasets.sumsin import sumsin
# Using the correct dtype for constatsns
my0 = numpy.asarray(0., dtype = theano.config.floatX)
my1 = numpy.asarray(1., dtype = theano.config.floatX)

def jobman(_options, channel = None):

    ################### PARSE INPUT ARGUMENTS #######################
    o = parse_input_arguments(_options,
                            'RNN_theano/rnn_sinsum001/RNN_sumsin.ini')
    ####################### DEFINE THE TASK #########################

    mode = Mode( linker = 'cvm_nogc', optimizer = 'fast_run')
    rng = numpy.random.RandomState(o['seed'])
    train_set = sumsin(
                      T             = o['task_T']
                    , steps         = o['task_steps']
                    , batches       = o['task_train_batches']
                    , batch_size    = o['task_train_batchsize']
                    , noise         = o['task_noise']
                    , rng           = rng
                    )

    valid_set = sumsin(
                      T              = o['task_T']
                    , steps          = o['task_steps']
                    , batches        = o['task_valid_batches']
                    , batch_size     = o['task_valid_batchsize']
                    , rng            = rng
                    )

    test_set = sumsin(
                      T             = o['task_T']
                    , steps         = o['task_steps']
                    , batches       = o['task_test_batches']
                    , batch_size    = o['task_test_batchsize']
                    , rng           = rng
                    )
    if o['wout_pinv'] :
        wout_set = sumsin(
                      T             = o['task_T']
                    , steps         = o['task_steps']
                    , batches       = o['task_wout_batches']
                    , batch_size    = o['task_wout_batchsize']
                    , noise         = o['task_wout_noise']
                    , rng           = rng
                    )

    ###################### DEFINE THE MODEL #########################

    def recurrent_fn( u_t, h_tm1, W_hh, W_ux, W_hy,b) :
        x_t = TT.dot(W_ux, u_t)
        h_t = TT.tanh( TT.dot(W_hh, h_tm1) + x_t + b)
        #y_t = TT.dot(W_hy, h_t)
        return h_t #, y_t

    u  = TT.matrix('u')
    if o['error_over_all']:
        t = TT.matrix('t')
    else:
        t  = TT.matrix('t')
    h0 = TT.vector('h0')
    b  = shared_shape( floatX(numpy.random.uniform(size=(o['nhid'],),
                                                   low =-o['Wux_properties']['scale'],
                                                   high= o['Wux_properties']['scale'])))
    alpha = TT.scalar('alpha')
    lr    = TT.scalar('lr')

    W_hh = init( o['nhid']
                , o['nhid']
                , 'W_hh'
                , o['Whh_style']
                , o['Whh_properties']
                , rng)

    W_ux_mask = numpy.ones((o['nhid'], train_set.n_ins), dtype =
                            theano.config.floatX)
    if o['Wux_mask_limit'] > 0:
        W_ux_mask[:o['Wux_mask_limit']] = 0.
    W_ux = init( o['nhid']
                , train_set.n_ins
                , 'W_ux'
                , o['Wux_style']
                , o['Wux_properties']
                , rng
                , mask = W_ux_mask)

    W_hy = init( train_set.n_outs
                , o['nhid']
                , 'W_hy'
                , o['Why_style']
                , o['Why_properties']
                , rng)
    h, _ = theano.scan(
        recurrent_fn
        , sequences = u
        , outputs_info = h0
        , non_sequences = [W_hh, W_ux, W_hy, b]
        , name = 'recurrent_fn'
        , mode = mode
        )
    y = TT.dot(W_hy, h.T)
    init_h =h.owner.inputs[0].owner.inputs[2]

    #h = theano.printing.Print('h',attrs=('shape',))(h)
    if o['error_over_all']:
        out_err = TT.mean((y-t)**2, axis = 1)
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
        if o['sum_h2'] > 0:
            proj = TT.join(0,proj[:o['sum_h2']],
                           TT.zeros_like(proj[o['sum_h2']:]))
        cost = TT.sum(proj*h[-1])

    z,gh = TT.grad(cost, [init_h, h])
    z.name = '__z__'
    #import GPUscan.ipdb; GPUscan.ipdb.set_trace()
    #z = z
    zsec = z[:-1] - gh
    if o['sum_h'] > 0:
        z2_1 = TT.sum(z[:,:o['sum_h']]**2, axis = 1)
        z2_2 = TT.sum(zsec[:,:o['sum_h']]**2, axis = 1)
    else:
        z2_1 = TT.sum(z**2, axis = 1)
        z2_2 = TT.sum(zsec**2, axis = 1)
    v1 = z2_2
    v2 = z2_1[1:]
    ## ## v2 = theano.printing.Print('v2')(v2)
    # floatX(1e-14)
    ratios = TT.switch(TT.ge(v2,1e-12), TT.sqrt(v1/v2), floatX(1))
    norm_0 = TT.ones_like(ratios[0])
    norm_t, _ = theano.scan(lambda x,y: x*y
                            , sequences = ratios
                            , outputs_info = norm_0
                            , name = 'jacobian_products'
                            , mode = mode
                           )
    norm_term = TT.sum(norm_t)
    if o['reg_cost'] == 'product':
        r = abs(TT.log(norm_t)).sum()
    elif o['reg_cost'] == 'each':
        part1 = abs(TT.log(ratios))
        part2 = TT.switch(TT.ge(v2,1e-12), part1, 1-v2)
        r = part2.sum()
    elif o['reg_cost'] == 'product2':
        ratios2 = TT.switch(TT.ge(z2[-1],1e-12), TT.sqrt(z2/z2[-1]),
                            floatX(1))
        r = abs(TT.log(ratios2)).sum()

    ratios = TT.switch(TT.ge(v2,1e-12), TT.sqrt(v1/v2), floatX(1e-12))[::-1]
    norm_0 = TT.ones_like(ratios[0])
    norm_t, _ = theano.scan(lambda x,y: x*y
                            , sequences = ratios
                            , outputs_info = norm_0
                            , name = 'jacobian_products'
                            , mode = mode
                           )
    norm_term = floatX(0.1)+TT.sum(norm_t)
    gu = TT.grad(y[-1].sum(), u)

    if o['opt_alg'] == 'sgd':
        get_updates = lambda p,e, up : ( sgd(p
                                           , e
                                           , lr      = lr
                                           , scale   =\
                                             TT.maximum( my1/norm_term,
                                                        floatX(0.01))
                                           , updates = up)[0]
                                        , [[],[],[TT.constant(0) for x in p]] )
    elif o['opt_alg'] == 'sgd_qn':
        get_updates = lambda p,e, up : sgd_qn(p
                                              , e
                                              , mylambda = floatX(o['mylambda'])
                                              , t0 = floatX(o['t0'])
                                              , skip = floatX(o['skip'])
                                              , scale =
                                              TT.maximum(my1/norm_term,
                                                         floatX(0.01))
                                              , lazy = o['lazy']
                                              , updates = up)

    if o['win_reg']:
        updates,why_extra = get_updates([W_hy], err, {})
        cost = err + alpha*r
        W_ux.name = 'W_ux'
        W_hh.name = 'W_hh'
        b.name = 'b'
        updates,extras = get_updates([W_ux, W_hh,b], cost, updates)
        updates[W_ux] = updates[W_ux]*W_ux_mask
        b_Why = why_extra[2][0]
        b_Wux = extras[2][0]
        b_Whh = extras[2][1]
        b_b   = extras[2][2]
    else:
        updates,extras1 = get_updates([W_hy, W_ux], err, {})
        updates[W_ux] = updates[W_ux]*W_ux_mask
        cost = err + alpha*r
        updates,extras2 = get_updates([W_hh,b], cost, updates)
        b_Why = extras1[2][0]
        b_Wux = extras1[2][1]
        b_Whh = extras2[2][0]
        b_b   = extras2[2][1]

    nhid = o['nhid']
    train_batchsize = o['task_train_batchsize']
    valid_batchsize = o['task_valid_batchsize']
    test_batchsize  = o['task_test_batchsize']
    wout_batchsize  = o['task_wout_batchsize']

    train_h0 = shared_shape(floatX(numpy.zeros((nhid,))))
    valid_h0 = shared_shape(floatX(numpy.zeros((nhid,))))
    test_h0  = shared_shape(floatX(numpy.zeros((nhid,))))
    wout_h0  = shared_shape(floatX(numpy.zeros((nhid,))))
    idx = TT.iscalar('idx')
    train_u, train_t = train_set(idx)
    u.tag.shape = copy.copy(train_u.tag.shape)
    t.tag.shape = copy.copy(train_t.tag.shape)
    train = theano.function([u, t, lr, alpha], [out_err, r, norm_term]
                            , updates = updates
                            , mode = mode
                            , givens = { h0: train_h0
                                       } )

    valid_u, valid_t = valid_set(idx)
    u.tag.shape = copy.copy(valid_u.tag.shape)
    t.tag.shape = copy.copy(valid_t.tag.shape)
    valid = theano.function([u,t], [out_err, r, norm_term]
                            , mode = mode
                            , givens = { h0: valid_h0
                                       } )

    test_u, test_t = test_set(idx)
    u.tag.shape = copy.copy(test_u.tag.shape)
    t.tag.shape = copy.copy(test_t.tag.shape)
    test = theano.function([u,t], [out_err
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
                                   , b_b
                                   , zsec
                                   , gh
                                  ]
                            , mode = mode
                            , givens = { h0: test_h0
                                       } )
    if o['wout_pinv']:
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
                , mode = mode
            )
            H_t = H_tm1 + TT.dot(h_t[-1], h_t[-1].T)
            Y_t = Y_tm1 + TT.dot(h_t[-1], t_t.T)
            return H_t, Y_t

        H_0 = shared_shape(numpy.zeros((o['nhid'], o['nhid'])
                                       , dtype = theano.config.floatX)
                            , name='H0')
        Y_0 = shared_shape(numpy.zeros((o['nhid'], 1)
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
            , mode = mode
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
    theano.printing.pydotprint(train, 'train.png', high_contrast=True,
                               with_ids= True)
    for idx,node in enumerate(train.maker.env.toposort()):
        if node.op.__class__.__name__ == 'Scan':
            theano.printing.pydotprint(node.op.fn,
                                       ('train%d_'%idx)+node.op.name,
                                       high_contrast = True,
                                       with_ids = True)

    theano.printing.pydotprint(train, 'valid.png', high_contrast=True,
                              with_ids = True)
    for idx,node in enumerate(train.maker.env.toposort()):
        if node.op.__class__.__name__ == 'Scan':
            theano.printing.pydotprint(node.op.fn,
                                       ('valid%d_'%idx)+node.op.name,
                                       high_contrast = True,
                                      with_ids = True)
    theano.printing.pydotprint(train, 'test.png', high_contrast=True,
                              with_ids = True)
    for idx,node in enumerate(train.maker.env.toposort()):
        if node.op.__class__.__name__ == 'Scan':
            theano.printing.pydotprint(node.op.fn,
                                       ('test%d_'%idx)+node.op.name,
                                       high_contrast = True,
                                      with_ids = True)
    if o['wout_pinv']:
        theano.printing.pydotprint(train, 'wout.png', high_contrast=True,
                                  with_ids = True)
        for idx,node in enumerate(train.maker.env.toposort()):
            if node.op.__class__.__name__ == 'Scan':
                theano.printing.pydotprint(node.op.fn,
                                       ('wout%d_'%idx)+node.op.name,
                                       high_contrast = True,
                                          with_ids= True)

    '''

    #import GPUscan.ipdb; GPUscan.ipdb.set_trace()
    #rval = valid(valid_set.data_u[0],valid_set.data_t[0])

    #################### DEFINE THE MAIN LOOP #######################


    data = {}
    fix_len = o['max_storage_numpy']#int(o['NN']/o['small_step'])
    avg_train_err  = numpy.zeros((o['small_step'],train_set.n_outs))
    avg_train_reg  = numpy.zeros((o['small_step'],))
    avg_train_norm = numpy.zeros((o['small_step'],))
    avg_valid_err  = numpy.zeros((o['small_step'],train_set.n_outs))
    avg_valid_reg  = numpy.zeros((o['small_step'],))
    avg_valid_norm = numpy.zeros((o['small_step'],))

    data['options'] = o
    data['train_err']  = -1*numpy.ones((fix_len,train_set.n_outs))
    data['valid_err']  = -1*numpy.ones((fix_len,train_set.n_outs))
    data['train_reg']  = -1*numpy.ones((fix_len,))
    data['valid_reg']  = -1*numpy.ones((fix_len,))
    data['train_norm'] = numpy.zeros((fix_len,))
    data['valid_norm'] = numpy.zeros((fix_len,))

    data['test_err']  = [None]*o['max_storage']
    data['test_idx']  = [None]*o['max_storage']
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
    data['stuff']     = []
    storage_exceeded  = False
    stop = False



    old_rval = numpy.inf
    patience = o['patience']
    n_train = o['task_train_batches']
    n_valid = o['task_valid_batches']
    n_test  = o['task_test_batches']
    n_test_runs = 0
    test_pos    = 0

    valid_set.refresh()
    test_set.refresh()
    kdx = 0
    lr_v  = floatX(o['lr'])
    alpha_v =floatX(o['alpha'])
    lr_f = 1
    if o['lr_scheme']:
        lr_f = o['lr_scheme'][1]/(o['NN'] - o['lr_scheme'][0])
    alpha_r = 1
    if o['alpha_scheme']:
        alpha_r = float(o['alpha_scheme'][1] - o['alpha_scheme'][0])

    st = time.time()
    if channel:
        try:
            channel.save()
        except:
            pass
    for idx in xrange(int(o['NN'])):
        if o['lr_scheme'] and idx > o['lr_scheme'][0]:
            lr_v = floatX(o['lr'] * 1./(1.+ (idx - o['lr_scheme'][0])*lr_f))
        if o['alpha_scheme']:
            if idx < o['alpha_scheme'][0]:
                alpha_v = floatX(0)
            elif idx < o['alpha_scheme'][1]:
                pos = 2.*(idx-o['alpha_scheme'][0])/alpha_r -1.
                alpha_v = floatX(numpy.exp(-pos**2/0.2)*o['alpha'])
            else:
                alpha_v = floatX(0)



        jdx = idx%o['small_step']
        avg_train_err[jdx,:] = 0
        avg_train_reg[jdx]   = 0
        avg_train_norm[jdx]  = 0

        avg_valid_err[jdx,:] = 0
        avg_valid_reg[jdx]   = 0
        avg_valid_norm[jdx]  = 0

        if o['wout_pinv'] and (idx%o['test_step'] == 0):
            wout_set.refresh()
            print ( '* Re-computing W_hy using closed-form '
                   'regularized wiener hopf formula')
            st_wout = time.time()
            wout(0)
            ed_wout = time.time()
            print '** It took ', ed_wout-st_wout,'secs'
            print '** Average weight', abs(W_hy.get_value(borrow=True)).mean()



        for k in xrange(o['task_train_batches']):
            s,t = train_set.get_slice()
            rval = train(s,t, lr_v, alpha_v)
            print '[',idx,'/',patience,'][',k,'/',n_train,'][train]', rval[0].mean(), \
                    rval[1], rval[2], numpy.max([(1./rval[2]), 0.01])*lr_v, alpha_v
            avg_train_err[jdx,:]  += rval[0]
            avg_train_reg[jdx]  += rval[1]
            avg_train_norm[jdx] += rval[2]
        print '**Epoch took', time.time() - st, 'secs'
        avg_train_err[jdx]  /= n_train
        avg_train_reg[jdx]  /= n_train
        avg_train_norm[jdx] /= n_train
        st = time.time()


        for k in xrange(n_valid):
            rval = valid(*valid_set.get_slice())
            print '[',idx,'/',patience,'][',k,'/',n_valid,'][valid]', rval[0].mean(), \
                    rval[1], rval[2]
            avg_valid_err[jdx]  += rval[0]
            avg_valid_reg[jdx]  += rval[1]
            avg_valid_norm[jdx] += rval[2]

        avg_valid_err[jdx]  /= n_valid
        avg_valid_reg[jdx]  /= n_valid
        avg_valid_norm[jdx] /= n_valid
        if idx >= o['small_step'] and idx%o['small_step'] == 0:
            kdx += 1
            if kdx >= o['max_storage_numpy']:
                kdx = o['max_storage_numpy']//3
                storage_exceeded = True

            data['steps'] = idx
            data['kdx']   = kdx
            data['storage_exceeded'] = storage_exceeded
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

                test_err  = []
                test_reg  = []
                test_norm = []


                for k in xrange(n_test):
                    rval = test(*  test_set.get_slice())
                    print '[',idx,'][',k,'/',n_test,'][test]',rval[0].mean()\
                        , rval[1], rval[2]
                    test_err   += [rval[0]]
                    test_reg   += [rval[1]]
                    test_norm  += [rval[2]]
                    test_z     = rval[7][:,:]
                    test_y     = rval[8][:,:]
                    test_h     = rval[9][:,:]
                    test_u     = rval[10][:,:]
                    test_gu    = rval[11][:,:]
                    test_t     = rval[12][:,:]
                data['test_idx'][test_pos]  = idx
                data['test_pos']            = test_pos
                data['y'][test_pos]         = test_y
                data['z'][test_pos]         = test_z
                data['t'][test_pos]         = test_t
                data['h'][test_pos]         = test_h
                data['u'][test_pos]         = test_u
                data['gu'][test_pos]        = test_gu
                data['test_err'][test_pos]  =  test_err
                data['test_reg'][test_pos]  =  test_reg
                data['test_norm'][test_pos] =  test_norm
                data['W_hh'][test_pos]      =  rval[3]
                data['W_ux'][test_pos]      =  rval[4]
                data['W_hy'][test_pos]      =  rval[5]
                data['b'][test_pos]         =  rval[6]
                data['b_hh'][test_pos]      =  rval[13]
                data['b_ux'][test_pos]      =  rval[14]
                data['b_hy'][test_pos]      =  rval[15]
                data['b_b'][test_pos]       =  rval[16]
                data['stuff'] += [(rval[17],rval[18])]
            cPickle.dump(data,
                open(os.path.join(
                    configs.results_folder(),
                    o['path'],'%s_backup.pkl'%o['name'])
                     ,'wb'))

        print '** ', avg_valid_err[jdx].mean(), ' < ', old_rval, ' ? '
        if avg_valid_err[jdx].mean() < old_rval :

            patience += o['patience_incr']
            if avg_valid_err[jdx].mean() < old_rval:




                test_err  = []
                test_reg  = []
                test_norm = []


                for k in xrange(n_test):
                    rval = test(* test_set.get_slice())
                    print '[',idx,'][',k,'/',n_test,'][test]',rval[0].mean()\
                        , rval[1], rval[2]
                    test_err   += [rval[0]]
                    test_reg   += [rval[1]]
                    test_norm  += [rval[2]]
                    test_z     = rval[7][:,:]
                    test_y     = rval[8][:,:]
                    test_h     = rval[9][:,:]
                    test_u     = rval[10][:,:]
                    test_gu    = rval[11][:,:]
                    test_t     = rval[12][:,:]
                data['test_idx'][test_pos]  = idx
                data['test_pos']            = test_pos
                data['y'][test_pos]         = test_y
                data['z'][test_pos]         = test_z
                data['t'][test_pos]         = test_t
                data['h'][test_pos]         = test_h
                data['u'][test_pos]         = test_u
                data['gu'][test_pos]        = test_gu
                data['test_err'][test_pos]  =  test_err
                data['test_reg'][test_pos]  =  test_reg
                data['test_norm'][test_pos] =  test_norm
                data['W_hh'][test_pos]      =  rval[3]
                data['W_ux'][test_pos]      =  rval[4]
                data['W_hy'][test_pos]      =  rval[5]
                data['b'][test_pos]         =  rval[6]
                data['b_hh'][test_pos]      =  rval[13]
                data['b_ux'][test_pos]      =  rval[14]
                data['b_hy'][test_pos]      =  rval[15]
                data['b_b'][test_pos]       =  rval[16]
                data['stuff'] += [(rval[17],rval[18])]

                cPickle.dump(data,
                    open(os.path.join(
                        configs.results_folder(),
                        o['path'],'%s.pkl'%o['name'])
                         ,'wb'))
                n_test_runs += 1
                test_pos    += 1
                if test_pos >= o['max_storage']:
                    test_pos = test_pos - o['go_back']
                if numpy.mean(test_err) < 5e-5:
                    patience = idx - 5
                    break

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

