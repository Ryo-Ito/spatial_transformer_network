import tensorflow as tf
from grid import batch_mgrid
from warp import batch_warp2d, batch_warp3d


def batch_displacement_warp2d(imgs, vector_fields):
    """
    warp images by free form transformation

    Parameters
    ----------
    imgs : tf.Tensor
        images to be warped
        [n_batch, xlen, ylen, n_channel]
    vector_fields : tf.Tensor
        [n_batch, 2, xlen, ylen]

    Returns
    -------
    output : tf.Tensor
    warped imagees
        [n_batch, xlen, ylen, n_channel]
    """
    n_batch = tf.shape(imgs)[0]
    xlen = tf.shape(imgs)[1]
    ylen = tf.shape(imgs)[2]

    grids = batch_mgrid(n_batch, xlen, ylen)

    T_g = grids + vector_fields
    output = batch_warp2d(imgs, T_g)
    return output


def batch_displacement_warp3d(imgs, vector_fields):
    """
    warp images by displacement vector fields

    Parameters
    ----------
    imgs : tf.Tensor
        images to be warped
        [n_batch, xlen, ylen, zlen, n_channel]
    vector_fields : tf.Tensor
        [n_batch, 3, xlen, ylen, zlen]

    Returns
    -------
    output : tf.Tensor
    warped imagees
        [n_batch, xlen, ylen, zlen, n_channel]
    """
    n_batch = tf.shape(imgs)[0]
    xlen = tf.shape(imgs)[1]
    ylen = tf.shape(imgs)[2]
    zlen = tf.shape(imgs)[3]

    grids = batch_mgrid(n_batch, xlen, ylen, zlen)

    T_g = grids + vector_fields
    output = batch_warp3d(imgs, T_g)
    return output
