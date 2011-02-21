#!/usr/bin/python

# Razvan Pascanu
'''
  Script to add more targeted jobs into the db
'''

from jobman.tools import DD
from jobman import sql
import numpy

# import types of architectures we want to use
import RNN


if __name__=='__main__':

    TABLE_NAME='rnn001'

    #db = sql.db('postgres://pascanur:he1enush@gershwin/pascanur_db/'+TABLE_NAME)
    db = sql.postgres_serial( user = 'pascanur',
            password='he1enush',
            host='colosse1',
            port = 5432,
            database='pascanur_db',
            table_prefix = TABLE_NAME)

    state = DD()

    ## DEFAULT VALUES ##
    state['configfile'] = '/home/rpascanu/workspace/RNN_theano/rnn001/RNN.ini'
    state['jobman.experiment'] = 'RNN.jobman'
    n_jobs = 0

    for n_jobs in xrange(512):
        n = numpy.random.rand()
        #if n > 0.5:
        state['opt_alg'] = 'sgd_qn'
        _lambda = -(numpy.random.randint(3)+4)
        state['mylambda'] = 10**_lambda
        start_lr = numpy.random.randint(3) + 1 + abs(_lambda)
        t0 = 10**start_lr
        state['t0'] = t0
        state['lr'] = 1./(t0*state['mylambda'])
        state['skip'] = numpy.random.randint(10) + 5
        state['lazy'] = True
        #else:
        #    state['opt_alg'] = 'sgd'
        #    lr = -(numpy.random.randint(2) + 3)
        #    state['lr'] = 10**lr
        #    state['lazy'] = False

        n = numpy.random.randint(6)
        if n == 0:
            state['whh_style'] = 'orthogonal'
            scale = (numpy.random.rand() + .2)/1.2
            state['whh_properties'] = "str:{'scale': %5.2f}"%scale
        elif n == 1:
            state['whh_style'] = 'esn'
            scale  = (numpy.random.rand() + .1)/1.1
            sparse = numpy.random.rand()*.5 +.01
            prop = "str:{'scale':%5.2f, 'sparsity' :%5.2f}"%(scale,sparse)
            state['whh_properties'] = prop
        else:
            state['whh_style'] = 'random'
            scale = (numpy.random.rand()*.2 + .0001)
            sparse = (numpy.random.rand()+.01)/1.01
            prop = "str:{'scale':%5.2f,'sparsity' :%5.2f}"%(scale,sparse)
            state['whh_properties'] = prop

        scale  = numpy.random.rand() + .0001
        sparse = (numpy.random.rand() +.01)/1.01
        prop = "str:{'scale':%5.2f,'sparsity' :%5.2f}"%(scale,sparse)
        state['why_properties'] = prop

        scale  = numpy.random.rand() + .0001
        sparse = (numpy.random.rand() +.01)/1.01
        prop = "str:{'scale':%5.2f,'sparsity' :%5.2f}"%(scale,sparse)
        state['wux_properties'] = prop

        state['nhid'] = numpy.random.randint(200) + 50
        nhid = state['nhid']
        n = numpy.random.randint(3)
        if n == 0:
            state['reg_projection'] = 'err'
            n2 = numpy.random.randint(4)
            if n2 == 0:
                state['sum_h'] = numpy.random.randint(nhid-5)+5
            else:
                state['sum_h'] = 0
            state['sum_h2'] = 0
        elif n == 1:

            n2 = numpy.random.randint(4)
            if n2 == 0:
                state['sum_h'] = numpy.random.randint(nhid-5)+5
            else:
                state['sum_h'] = 0
            state['sum_h2'] = 0
            state['reg_projection'] = 'h[-1]'
        else:

            n2 = numpy.random.randint(4)
            if n2 == 0:
                state['sum_h'] = numpy.random.randint(nhid-5)+5
                state['sum_h2'] = 0
            elif n2 == 1:
                state['sum_h2'] = numpy.random.randint(nhid-5)+5
                state['sum_h'] = 0
            else:
                state['sum_h'] = 0
                state['sum_h2'] = 0
            state['reg_projection'] = 'random'

        n = numpy.random.randint(3)
        if n == 0:
            state['reg_cost'] = 'product'
        elif n == 1:
            state['reg_cost'] = 'each'
        else:
            state['reg_cost'] = 'product2'

        n = numpy.random.randint(2)
        if n == 0:
            state['win_reg'] = 'str:True'
        else:
            state['win_reg'] = 'str:False'

        n = numpy.random.randint(3)
        if n == 0:
            state['wout_pinv'] = 'str:True'
        else:
            state['wout_pinv'] = 'str:False'

        n = numpy.random.randint(5)
        if n == 0:
            state['alpha'] = 0.
        else:
            lmbd = -numpy.random.randint(5)
            state['alpha'] = ((numpy.random.rand()*5.+1.)/6.)*(10**lmbd)
        state['seed'] = numpy.random.randint(2000)
        n = numpy.random.randint(3)
        if n == 0 :
            state['n_outs'] = 1
        elif n == 1:
            state['n_outs'] = 3
        else:
            state['n_outs'] = 20

        expo = -numpy.random.randint(3)-1
        val = numpy.random.rand()*(10**expo)
        state['wiener_lambda'] = val
        state['test_step'] = 10**numpy.random.randint(6)

        n = numpy.random.randint(2)
        if n == 0 :
            state['style'] = 'continous'
        else:
            state['style'] = 'random_pos'

        n = numpy.random.randint(7)
        if n == 0:
            expo = -numpy.random.randint(3) -1
            val = numpy.random.rand()*(10**expo)
            state['task_noise'] = val
        else:
            state['task_noise'] = 0.

        n = numpy.random.randint(7)
        if n == 0:
            expo = -numpy.random.randint(3) -1
            val = numpy.random.rand()*(10**expo)
            state['task_wout_noise'] = val
        else:
            state['task_wout_noise'] = 0.


        state['name'] = 'rnn_%03d'%n_jobs
        sql.add_experiments_to_db(
            [state], db, verbose=1, force_dup =True)


    print 'N_jobs ', n_jobs, TABLE_NAME
