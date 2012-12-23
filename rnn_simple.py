import numpy, time
import theano
import theano.tensor as TT
import pickle, sys
import pickle, os

def jobman(options, channel):
    ####### CONFIGURE PARAMETERS
    floatX = theano.config.floatX
    if channel is not None:
        channel.save()
    n_hidden = options['nhid']
    rng = numpy.random.RandomState(options['seed'])
    profile = 0
    lr = theano.shared(numpy.array(float(options['lr']), dtype=floatX))
    max_val =options['cutoff']


    ###### LOAD DATASET
    print 'Generating data'
    T0 = options['T0']
    T = T0
    nin = 6
    nout = 4

    def generate_fresh_data():
        train_x = numpy.zeros((options['T0'], options['bs'], 6))
        train_y = numpy.zeros((options['bs'], 4))
        symbols = numpy.random.randint(4, size=(options['T0'], options['bs']))
        for bdx in xrange(options['bs']):
            t1 = numpy.random.randint(low=options['T0']//10, high =
                                      options['T0'] // 5)
            t2 = numpy.random.randint(low =options['T0']//2, high =
                                      6*options['T0'] // 10)
            orderCase = numpy.random.randint(4)
            if orderCase == 0:
                symbols[t1, bdx] = 4
                symbols[t2, bdx] = 4
            elif orderCase == 1:
                symbols[t1, bdx] = 4
                symbols[t2, bdx] = 5
            elif orderCase == 2:
                symbols[t1, bdx] = 5
                symbols[t2, bdx] = 4
            else:
                symbols[t1, bdx] = 5
                symbols[t2, bdx] = 5
            for t in xrange(options['T0']):
                train_x[t,bdx, symbols[t, bdx]] = 1
            train_y[bdx, orderCase] = 1
        train_x[:4] *= .3
        train_x = train_x.astype(floatX)
        train_y = train_y.astype(floatX)
        return train_x, train_y


    ########## INITIALIZE MATRISES
    W_uh = rng.uniform(size=(nin, n_hidden),
                       low = -.1,
                       high=  .1)
    W_uh = W_uh.astype(floatX)

    # Hidden weights are initialized to a sparse matrix with a fixed largest
    # eigenvalues
    trials    = 0
    success = False
    while not success:
        try:
            values = rng.uniform(low = -1, high = 1,
                                 size=(n_hidden*n_hidden,))
            sparsity = 0.9
            if 'sparsity' in options:
                sparsity = options['sparsity']
            positions = rng.permutation(n_hidden*n_hidden)
            limit = int(n_hidden*n_hidden*sparsity)
            if n_hidden < 30 :
                limit = n_hidden*n_hidden-n_hidden
            values[positions[:limit]] = 0.
            values = values.reshape((n_hidden,n_hidden))
            maxval = numpy.max(numpy.abs(numpy.linalg.eigvals(values)))
            scale = 0.5
            if 'scale' in options:
                scale = options['scale']
            values = values * scale / maxval
            success = True
        except:
            print 'ESN weights generation, trail', trials
            trials += 1
            if trials > 20:
                raise ValueError('Could not generate ESN weights')
    W_hh = values.astype(floatX)

    W_hy = rng.uniform(size=(n_hidden, nout),
                       low= -.01,
                       high=.01)
    W_hy = W_hy.astype(floatX)
    b_hh = numpy.zeros((n_hidden,), dtype=floatX)
    b_hy = numpy.zeros((nout,), dtype=floatX)
    W_uh = theano.shared(W_uh, 'W_uh')
    W_hh = theano.shared(W_hh, 'W_hh')
    W_hy = theano.shared(W_hy, 'W_hy')
    b_hh = theano.shared(b_hh, 'b_hh')
    b_hy = theano.shared(b_hy, 'b_hy')
    ########### DEFINE TRAINING FUNCTION
    u = TT.matrix('u')
    t = TT.matrix('t')
    h0_tm1 = TT.alloc(numpy.array(0, dtype=floatX), options['bs'], n_hidden)
    # Project input through input weights
    u_proj = TT.dot(u, W_uh)
    p_u_proj = u_proj.reshape((options['T0'], options['bs'], n_hidden))
    # Define the recurrent network
    def recurrent_fn(u_t, h_tm1):
        h_t = TT.tanh(TT.dot(h_tm1, W_hh) + u_t + b_hh)
        return h_t

    h, _ = theano.scan(
        recurrent_fn,
        sequences = p_u_proj,
        outputs_info = [h0_tm1],
        name = 'rfn',
        profile=0,
        mode = theano.Mode(linker='cvm'))
    # Compute error
    y = TT.dot(h.reshape((options['T0']*options['bs'], n_hidden)), W_hy) +b_hy
    y = TT.nnet.softmax(y).reshape((options['T0'], options['bs'], nout))
    cost = -(TT.xlogx.xlogy0(t, y[-1]) +
             TT.xlogx.xlogy0(1-t, 1-y[-1]))
    cost = cost.sum(1).mean(0)
    err2 = TT.neq(TT.argmax(y[-1], axis=1), TT.argmax(t, axis=1)).sum()
    # Compute gradients
    gW_hh, gW_uh, gW_hy, gb_hh, gb_hy = TT.grad(cost, [W_hh, W_uh, W_hy, b_hh, b_hy])
    norm_theta = TT.sqrt((gW_hh**2).sum() +
                      (gW_uh**2).sum() +
                      (gW_hy**2).sum() +
                      (gb_hh**2).sum() +
                      (gb_hy**2).sum())
    c = options['cutoff']
    gW_hh = TT.switch(norm_theta > c, c*gW_hh/norm_theta, gW_hh)
    gW_uh = TT.switch(norm_theta > c, c*gW_uh/norm_theta, gW_uh)
    gW_hy = TT.switch(norm_theta > c, c*gW_hy/norm_theta, gW_hy)
    gb_hh = TT.switch(norm_theta > c, c*gb_hh/norm_theta, gb_hh)
    gb_hy = TT.switch(norm_theta > c, c*gb_hy/norm_theta, gb_hy)


    eval_step = theano.function([u,t], [cost, err2])
    train_step = theano.function([u,t],cost,
                                 name='rec_fn',
                                 mode=theano.Mode(linker='cvm'),
                                 profile=0,
                            updates=[(W_hh, W_hh - lr*gW_hh),
                                     (W_uh, W_uh - lr*gW_uh),
                                     (W_hy, W_hy - lr*gW_hy),
                                     (b_hh, b_hh - lr*gb_hh),
                                     (b_hy, b_hy - lr*gb_hy)])



    print 'Starting to train'
    old_eval = numpy.inf
    best_score = numpy.inf
    patience = 5
    cont = True
    n = -1
    consv = 0
    solved = 0
    options['solved'] =0
    start_time = time.time()
    while lr.get_value() > 1e-8 and cont:
        train_x, train_y = generate_fresh_data()
        n = n+1
        st = time.time()
        tr_cost = train_step(train_x.reshape((options['T0']*options['bs'],-1)), train_y)

        valid_x, valid_y = generate_fresh_data()
        cost0, cost1 = eval_step(valid_x.reshape((options['T0']*options['bs'],-1)), valid_y)
        ed = time.time()
        if cost0 < best_score:
            best_score = cost0
            options['bestvalid0'] = float(cost0)
            options['bestvalid1'] = float(cost1)
            options['besttime'] = float(time.time() - start_time)
            options['beststep'] = int(n)
        if cost1 == 0 and numpy.isfinite(cost0):
            cont = False
            print 'Train ',n,':',tr_cost,\
                      'validation', \
                      cost0, cost1,\
                      'best validation', best_score,\
                        'sr', \
                    numpy.max(abs(numpy.linalg.eigvals(
                        W_hh.get_value()))), \
                      'time', ed-st, 'sec', \
                      'patience', patience
            solved=1
            print 'problem solved'
        if abs(cost0) > abs(old_eval*5.0) or not numpy.isfinite(cost0):
            print 'Decreasing lr', cost0, old_eval, lr.get_value()
            lr.set_value(lr.get_value()/2.)
        else:
            patience = n + 100
        if n > patience or n > 1e5: #score > old_eval:
            cont = False
        old_eval = cost0
        options['lr'] = float(lr.get_value())
        options['train'] = float(tr_cost)
        options['valid'] =  float(cost0)
        options['time'] = float(time.time() - start_time)
        options['step'] = int(n)
        if n % 100 == 0:
            print 'Train ',n,':',tr_cost,\
                      'validation', \
                      cost0, cost1,\
                      'best validation', best_score,\
                        'sr', \
                    numpy.max(abs(numpy.linalg.eigvals(
                        W_hh.get_value()))), \
                      'time', ed-st, 'sec', \
                      'patience', patience
        if n % 1e4 == 0 and channel is not None:
            channel.save()
    if solved:
        options['solved'] = 1
    else:
        options['solved'] = 0
    if channel is not None:
        channel.save()


if __name__=='__main__':
    options = {}
    options['nhid'] = 100
    options['seed'] = 123
    options['sparsity'] = .8
    options['scale'] = 1.
    options['lr'] = 1e-3
    options['T0'] = 35
    options['bs'] = 50
    options['cutoff'] = 15.
    jobman(options, None)
