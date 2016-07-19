import tensorflow as tf


def mgrid2d(xlen, ylen, low=-1., high=1.):
    """
    create orthogonal grid
    similart to np.mgrid

    Parameters
    ----------
    xlen : int
        number of points on x axis
    ylen : int
        number of points on y axis
    low : float
        minimum coordinates
    high : float
        maximum coordinates

    Returns
    -------
    grid : tf.Tensor
        [2, xlen, ylen]
    """
    low = tf.to_float(low)
    high = tf.to_float(high)
    x_coords = tf.linspace(low, high, xlen)
    y_coords = tf.linspace(low, high, ylen)
    zeros = tf.zeros([xlen, ylen])
    x_t = zeros + tf.reshape(x_coords, [-1, 1])
    y_t = zeros + tf.reshape(y_coords, [1, -1])

    grid = tf.concat(0, [x_t, y_t])
    return tf.reshape(grid, [2, xlen, ylen])


def batch_mgrid2d(xlen, ylen, n_batch=1, low=-1., high=1.):
    """
    create orthogonal grids
    similart to np.mgrid

    Parameters
    ----------
    xlen : int
        number of points for x axis
    ylen : int
        number of points for y axis
    n_batch : int
        number of grids to create
    low : float
        minimum value of coordinates
    high : float
        maximum value of coordinates

    Returns
    -------
    grid : tf.Tensor
        [n_batch, 2, xlen, ylen]
    """
    grid = mgrid2d(xlen, ylen, low, high)
    grid = tf.expand_dims(grid, 0)
    grid = tf.reshape(grid, [-1])
    grid = tf.tile(grid, [n_batch])
    grid = tf.reshape(grid, [n_batch, 2, xlen, ylen])
    return grid


def mgrid3d(xlen, ylen, zlen, low=-1., high=1.):
    """
    almost equivalent to np.mgrid[:xlen, :ylen, :zlen]

    Parameters
    ----------
    xlen : int
        number of points on x axis
    ylen : int
        number of points on y axis
    zlen : int
        number of points on z axis
    low : float
        minimum value of coordinates
    high : float
        maximum value of coordinates

    Returns
    -------
    grid : tf.Tensor
        [3, xlen, ylen, zlen]
    """
    low = tf.to_float(low)
    high = tf.to_float(high)
    x_coords = tf.linspace(low, high, xlen)
    y_coords = tf.linspace(low, high, ylen)
    z_coords = tf.linspace(low, high, zlen)
    zeros = tf.zeros([xlen, ylen, zlen])
    x_t = zeros + tf.reshape(x_coords, [-1, 1, 1])
    y_t = zeros + tf.reshape(y_coords, [1, -1, 1])
    z_t = zeros + tf.reshape(z_coords, [1, 1, -1])

    grid = tf.concat(0, [x_t, y_t, z_t])
    return tf.reshape(grid, [3, xlen, ylen, zlen])


def batch_mgrid3d(xlen, ylen, zlen, n_batch, low=-1., high=1.):
    """
    creates n_batch orthogonal grids

    Parameters
    ----------
    xlen : int
        number of points on x axis
    ylen : int
        number of points on y axis
    zlen : int
        number of points on z axis
    n_batch : int
        number of grids to create
    low : float
        minimum value of coordinates
    high : float
        maximum value of coordinates

    Returns
    -------
    grids : tf.Tensor
        [n_batch, 3, xlen, ylen, zlen]
    """
    grid = mgrid3d(xlen, ylen, zlen, low, high)
    grid = tf.expand_dims(grid, 0)
    grid = tf.reshape(grid, [-1])
    grid = tf.tile(grid, [n_batch])
    grid = tf.reshape(grid, [n_batch, 3, xlen, ylen, zlen])
    return grid
