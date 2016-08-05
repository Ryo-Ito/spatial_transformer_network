import tensorflow as tf
from grid import batch_mgrid
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

    grids = batch_mgrid(n_batch, xlen, ylen)
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

    grids = batch_mgrid(n_batch, xlen, ylen, zlen)
    grids = tf.reshape(grids, [n_batch, 3, -1])

    T_g = tf.batch_matmul(matrix, grids) + t
    T_g = tf.reshape(T_g, [n_batch, 3, xlen, ylen, zlen])
    output = warp3d(imgs, T_g)
    return output


if __name__ == '__main__':
    """
    for test

    the result will be

    the original image
    [[  0.   1.   2.   3.   4.]
     [  5.   6.   7.   8.   9.]
     [ 10.  11.  12.  13.  14.]
     [ 15.  16.  17.  18.  19.]
     [ 20.  21.  22.  23.  24.]]

    identity warped
    [[  0.   1.   2.   3.   4.]
     [  5.   6.   7.   8.   9.]
     [ 10.  11.  12.  13.  14.]
     [ 15.  16.  17.  18.  19.]
     [ 20.  21.  22.  23.  24.]]

    zoom in warped
    [[  6.    6.5   7.    7.5   8. ]
     [  8.5   9.    9.5  10.   10.5]
     [ 11.   11.5  12.   12.5  13. ]
     [ 13.5  14.   14.5  15.   15.5]
     [ 16.   16.5  17.   17.5  18. ]]
    """
    import numpy as np
    img = tf.to_float(np.arange(25).reshape(1, 5, 5, 1))
    identity_matrix = tf.to_float([1, 0, 0, 0, 1, 0])
    zoom_in_matrix = identity_matrix * 0.5
    identity_warped = affine_warp2d(img, identity_matrix)
    zoom_in_warped = affine_warp2d(img, zoom_in_matrix)
    with tf.Session() as sess:
        print sess.run(img[0, :, :, 0])
        print sess.run(identity_warped[0, :, :, 0])
        print sess.run(zoom_in_warped[0, :, :, 0])
