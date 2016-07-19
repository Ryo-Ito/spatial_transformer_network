import tensorflow as tf
from grid import batch_mgrid2d, batch_mgrid3d
from warp import warp2d, warp3d


def affine_warp2d(imgs, theta):
    """
    affine transforms 2d images

    Parameters
    ----------
    imgs : tf.Tensor
        images to be warped
        [n_batch, xlen, ylen, n_channel]
    theta : tf.Tensor
        parameters of affine transformation
        [n_batch, 6]

    Returns
    -------
    output : tf.Tensor
        warped images
        [n_batch, xlen, ylen, n_channel]
    """
    n_batch = tf.shape(imgs)[0]
    xlen = tf.shape(imgs)[1]
    ylen = tf.shape(imgs)[2]
    theta = tf.reshape(theta, [-1, 2, 3])
    matrix = tf.slice(theta, [0, 0, 0], [-1, -1, 2])
    t = tf.slice(theta, [0, 0, 2], [-1, -1, -1])

    grids = batch_mgrid2d(xlen, ylen, n_batch=n_batch)
    coords = tf.reshape(grids, [n_batch, 2, -1])

    T_g = tf.batch_matmul(matrix, coords) + t
    T_g = tf.reshape(T_g, [n_batch, 2, xlen, ylen])
    output = warp2d(imgs, T_g)
    return output


def affine_warp3d(imgs, theta):
    """
    affine transforms 3d images

    Parameters
    ----------
    imgs : tf.Tensor
        images to be warped
        [n_batch, xlen, ylen, zlen, n_channel]
    theta : tf.Tensor
        parameters of affine transformation
        [n_batch, 12]

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
    theta = tf.reshape(theta, [-1, 3, 4])
    matrix = tf.slice(theta, [0, 0, 0], [-1, -1, 3])
    t = tf.slice(theta, [0, 0, 3], [-1, -1, -1])

    grids = batch_mgrid3d(xlen, ylen, zlen, n_batch)
    grids = tf.reshape(grids, [n_batch, 3, -1])

    T_g = tf.batch_matmul(matrix, grids) + t
    T_g = tf.reshape(T_g, [n_batch, 3, xlen, ylen, zlen])
    output = warp3d(imgs, T_g)
    return output
