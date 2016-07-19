import tensorflow as tf
from grid import batch_mgrid2d, batch_mgrid3d
from warp import warp2d, warp3d


def displacement_warp2d(imgs, vector_fields):
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

    grids = batch_mgrid2d(xlen, ylen, n_batch=n_batch)

    T_g = grids + vector_fields
    output = warp2d(imgs, T_g)
    return output


def displacement_warp3d(imgs, vector_fields):
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

    grids = batch_mgrid3d(xlen, ylen, zlen, n_batch=n_batch)

    T_g = grids + vector_fields
    output = warp3d(imgs, T_g)
    return output
