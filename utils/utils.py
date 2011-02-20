"""
 Some utils functions closely related to RNN model.
"""

__authors__ = "Razvan Pascanu"
__copyright__ = "(c) 2011, Universite de Montreal"
__contact__ = "Razvan Pascanu <r.pascanu@gmail>"
import ConfigParser, cPickle, gzip, numpy, time, optparse, os

import theano
import theano.tensor as TT

def plot_scan_inner_graphs(fn, path, base_name = '', **kwargs):
    for o in fn.maker.env.toposort():
        if o and o.op.__class__.__name__=='Scan':
            if o.op.name:
                file_name = os.path.join(
                    path
                    , '%s_%s.png'%(base_name,o.op.name))
            else:
                file_name = os.path.join(path, '%s_scan.png'%base_name)

            theano.printing.pydotprint( o.op.fn, file_name, **kwargs)


def parse_input_arguments(_options, default = 'mainrc'):

    config = ConfigParser.ConfigParser()
    config.optionxform = str
    if 'configfile' in _options and _options['configfile'] is not None:
        config.readfp(open(_options['configfile']))
    else:
        config.readfp(open(default))

    o = dict( config.items('global'))

    for k,v in o.iteritems():
        if k in _options and _options[k] is not None:
            o[k] = _options[k]
            print k,v, _options[k]
        else:
            o[k] = eval(v)
    for k,v in _options.iteritems():
        if k not in o:
            o[k] = v
    return o

def floatX(x):
    return numpy.asarray(x, dtype = theano.config.floatX)
