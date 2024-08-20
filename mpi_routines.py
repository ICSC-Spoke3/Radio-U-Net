from inputparameters import *
if USE_MPI == 1:
   from mpi4py import MPI
import json
import numpy as np
import tensorflow as tf
import itertools


def get_cuda_devices(rank):
    cuda.init()
    ctx = cuda.Device(rank).make_context()
    ngpus = ctx.get_device().count()
    ctx.pop()
    ctx.detach()
    return ngpus


class mpi_dum:
    rank = 0
    size = 1

def read_json(hyper_file):

    with open(hyper_file) as json_file:
        json_data = json.load(json_file)

    return json_data

def distribute_parameters(json_data, comm=None, restart_file=None):

    hyperparam_keys = ['nbatch', 'num_epoch', 'eta_input', 'drop_out', 'deepmodel']
    if USE_MPI == 0:
       comm = mpi_dum()

    if comm.rank == 0:
        if restart_file is not None:
            with open(restart_file) as f:
                hyperparam_dict = json.load(f)
            hyperparam_values = [list(h.values()) for h in hyperparam_dict]
        else:
            hyperparam_values = list(itertools.product(*[json_data[d]
                                     for d in hyperparam_keys]))

        hyperparam_values = list(np.array_split(hyperparam_values,
                                 comm.size))
        hyper_index = np.array([len(l) for l in hyperparam_values])
    else:
        hyperparam_values = None
        hyper_index = None

    if USE_MPI == 1:
       hyperparam_values = comm.scatter(hyperparam_values, root=0)
       hyper_index = comm.bcast(hyper_index, root=0)

    hyperparam = []

    if USE_MPI == 1:
      for h in hyperparam_values:
         hyperparam.append(dict(zip(hyperparam_keys, h)))
    else:
       for h1 in hyperparam_values:
         for h in h1:
           hyperparam.append(dict(zip(hyperparam_keys, h)))

    param_dict = {'hyperparam': hyperparam,
                  'hyper_index': hyper_index}



    return param_dict

def set_gpus():

    ngpus = 4
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    #config.gpu_options.visible_device_list = str(divmod(MPI.COMM_WORLD.rank,ngpus)[1])
    #config.log_device_placement=True
    #config.allow_soft_placement=True
    config.gpu_options.allow_growth = True
    return config


"""

def set_horovod(hvd_rank, intra_comm):

    ngpus = get_cuda_devices(hvd_rank)
    hvd.init(intra_comm)
    assert hvd.mpi_threads_supported()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.visible_device_list = str(divmod(MPI.COMM_WORLD.rank,
                                                 ngpus)[1])
    sess = tf.Session(config=config)
    K.set_session(sess)


def set_intra_comm(nreplica):

    world_rank = MPI.COMM_WORLD.rank
    world_size = MPI.COMM_WORLD.size
    ranks_array = np.array_split(np.arange(world_size), nreplica)
    inter_array = np.array([np.argsort(i) for i in ranks_array]).flatten()

    for i, j in enumerate(ranks_array):
        if world_rank in j:
            color_intra = i

    color_inter = inter_array[world_rank]

    intra_comm = MPI.COMM_WORLD.Split(color_intra, world_rank)
    inter_comm = MPI.COMM_WORLD.Split(color_inter, world_rank)

    return intra_comm, inter_comm
"""
