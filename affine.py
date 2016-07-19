import tensorflow as tf
from grid import batch_mgrid2d, batch_mgrid3d
from warp import warp2d, warp3d


def affine_warp2d(imgs, matrix, bias):
    """
    affine transforms 2d images

    Parameters
    ----------
    imgs : tf.Tensor
        images to be warped
        [n_batch, xlen, ylen, n_channel]
    matrix : tf.Tensor
        linear transformation
        [n_batch, 2, 2]
    bias : tf.Tensor
        translation
        [n_batch, 2]

    Returns
    -------
    output : tf.Tensor
        warped images
        [n_batch, xlen, ylen, n_channel]
    """
    n_batch = tf.shape(imgs)[0]
    xlen = tf.shape(imgs)[1]
    ylen = tf.shape(imgs)[2]
    matrix = tf.to_float(matrix)
    bias = tf.to_float(bias)

    grids = batch_mgrid2d(xlen, ylen, n_batch=n_batch)
    coords = tf.reshape(grids, [n_batch, 2, -1])

    T_g = tf.batch_matmul(matrix, coords) + tf.expand_dims(bias, 2)
    T_g = tf.reshape(T_g, [n_batch, 2, xlen, ylen])
    output = warp2d(imgs, T_g)
    return output


def affine_warp3d(imgs, matrix, bias):
    """
    affine transforms 3d images

    Parameters
    ----------
    imgs : tf.Tensor
        images to be warped
        [n_batch, xlen, ylen, zlen, n_channel]
    matrix : tf.Tensor
        linear transformation
        [n_batch, 3, 3]
    bias : tf.Tensor
        translation
        [n_batch, 3]

    Returns
    -------
    output : tf.Tensor
        warped images
        [n_batch, xlen, ylen, zlen, n_channel]
    """
    n_batch = tf.shape(imgs)[0]
    xlen = tf.shape(imgs)[1]
    ylen = tf.shape(imgs)[2]
    zlen = tf.shape(imgs)[3]
    matrix = tf.to_float(matrix)
    bias = tf.to_float(bias)

    grids = batch_mgrid3d(xlen, ylen, zlen, n_batch)
    grids = tf.reshape(grids, [n_batch, 3, -1])

    T_g = tf.batch_matmul(matrix, grids) + tf.expand_dims(bias, 2)
    T_g = tf.reshape(T_g, [n_batch, 3, xlen, ylen, zlen])
    output = warp3d(imgs, T_g)
    return output
