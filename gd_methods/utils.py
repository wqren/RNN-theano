import numpy
import cPickle
import theano
import theano.tensor as T

from theano.scalar import BinaryScalarOp, upcast_out, int_types, float_types
class TrueDivSpecial(BinaryScalarOp):
    nin = 3
    def output_types(self, types):
        if all(t.dtype.startswith('int') for t in types):
            return [float64]
        else:
            return super(TrueDivSpecial, self).output_types(types)
    def impl(self, x, y, l):
        x = numpy.asarray(x)
        y = numpy.asarray(y)
        if str(x.dtype).startswith('int') and str(y.dtype).startswith('int'):
            if y == 0.:
                return l
            else:
                return float(x) / y
        else:
            if y == 0:
                return l
            else:
                return x / y
    def c_code(self, node, name, (x, y, l), (z, ), sub):
        if node.inputs[0].type in int_types and node.inputs[1].type in int_types:
            return "if (%(y)s!=0.) {%(z)s = ((double)%(x)s) / %(y)s;} else {%(z)s = %(l)s;}" % locals()
        return "if (%(y)s!=0.) {%(z)s = %(x)s / %(y)s;} else {%(z)s = %(l)s;}" % locals()
    def grad(self, (x, y, l), (gz, )):
        raise NotImplementedError()
        
true_div_special = TrueDivSpecial(upcast_out, name = 'true_div_special')

def clone(i, o, replaced_inputs = []):
    """ WRITEME

    :type i: list
    :param i: input L{Variable}s
    :type o: list
    :param o: output L{Variable}s
    :type copy_inputs: bool
    :param copy_inputs: if True, the inputs will be copied (defaults to False)

    Copies the subgraph contained between i and o and returns the
    outputs of that copy (corresponding to o).
    """
    equiv = clone_get_equiv(i, o, replaced_inputs)
    return equiv
    #return [equiv[input] for input in i], [equiv[output] for output in o]


def clone_get_equiv(i, o, replaced_inputs = []):
    """ WRITEME

    :type i: list
    :param i: input L{Variable}s
    :type o: list
    :param o: output L{Variable}s
    :type copy_inputs_and_orphans: bool
    :param copy_inputs_and_orphans:
        if True, the inputs and the orphans will be replaced in the cloned graph by copies
        available in the equiv dictionary returned by the function (copy_inputs_and_orphans
        defaults to True)

    :rtype: a dictionary
    :return:
        equiv mapping each L{Variable} and L{Op} in the graph delimited by i and o to a copy
        (akin to deepcopy's memo).
    """

    from theano.gof.graph import io_toposort
    from theano.gof import Container
    from copy import deepcopy

    copy_inputs_and_orphans = True
    d = {}
    for input in i:
        if input in replaced_inputs:
            cpy = input.clone()
            # deep-copying the container, otherwise the copied input's container will point to the same place
            cont = input.container
            cpy.container = Container(cpy,
                    storage=[input.type.filter(deepcopy(cpy.value), strict=cont.strict, allow_downcast=cont.allow_downcast)],
                    readonly=cont.readonly,
                    strict=cont.strict,
                    allow_downcast=cont.allow_downcast)
            cpy.owner = None
            cpy.index = None
            d[input] = cpy
        else:
            d[input] = input

    for apply in io_toposort(i, o):
        for input in apply.inputs:
            if input not in d:
                if copy_inputs_and_orphans and input in replaced_inputs:
                    # TODO: not quite sure what to do here
                    cpy = input.clone()
                    d[input] = cpy
                else:
                    d[input] = input

        new_apply = apply.clone_with_new_inputs([d[i] for i in apply.inputs])
        d[apply] = new_apply
        for output, new_output in zip(apply.outputs, new_apply.outputs):
            d[output] = new_output

    for output in o:
        if output not in d:
            d[output] = output.clone()

    return d


def load_data(dataset):
    ''' Loads the dataset

    :type dataset: string
    :param dataset: the path to the dataset (here MNIST)
    '''

    #############
    # LOAD DATA #
    #############
    print '... loading data'

    # Load the dataset 
    f = open(dataset,'rb')
    train_set_x, train_set_y, test_set_x, test_set_y = cPickle.load(f)
    f.close()


    def shared_dataset(data_xy):
        """ Function that loads the dataset into shared variables
        
        The reason we store our dataset in shared variables is to allow 
        Theano to copy it into the GPU memory (when code is run on GPU). 
        Since copying data into the GPU is slow, copying a minibatch everytime
        is needed (the default behaviour if the data is not in a shared 
        variable) would lead to a large decrease in performance.
        """
        data_x, data_y = data_xy
        shared_x = theano.shared(numpy.asarray(data_x, dtype=theano.config.floatX))
        shared_y = theano.shared(numpy.asarray(data_y, dtype=theano.config.floatX))
        # When storing data on the GPU it has to be stored as floats
        # therefore we will store the labels as ``floatX`` as well
        # (``shared_y`` does exactly that). But during our computations
        # we need them as ints (we use labels as index, and if they are 
        # floats it doesn't make sense) therefore instead of returning 
        # ``shared_y`` we will have to cast it to int. This little hack
        # lets ous get around this issue
        return shared_x, shared_y

    test_set_x,  test_set_y  = shared_dataset((test_set_x,test_set_y))
    train_set_x, train_set_y = shared_dataset((train_set_x,train_set_y))

    rval = [(train_set_x, train_set_y), (test_set_x, test_set_y)]
    return rval



