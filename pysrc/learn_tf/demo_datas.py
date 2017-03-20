import numpy as np

# generate circle-data
def circle_data(nb_points=50):
    '''
    helper function to generate circle two types of data
        one in circle center ((0,0), 2) labeled 0
        one in area of circle center ((0,0),5) - ((0,0),3) labeled 1
    :param nb_points: an interger controls how many points per label
    :return:
            inputs: 2d-points samples
            labels: points' labels
    '''
    inputs  = np.zeros([2*nb_points,2])
    labels = np.zeros([2*nb_points], dtype=np.int32)

    for i in range(nb_points):
        r,t = np.random.uniform(0., 2.), np.random.uniform(0., 2.*np.pi)
        inputs[i] = [r*np.sin(t), r*np.cos(t)]
        r,t = np.random.uniform(3., 5.), np.random.uniform(0., 2.*np.pi)
        inputs[i+nb_points] = [r*np.sin(t), r*np.cos(t)]
        labels[i+nb_points] = 1

    return inputs, labels

# generate 2d-curve
def curve2d_data(nb_points=50):
    '''
    helper function to generate 2 curves one labeled 0, the other labeled 1
    :param nb_points: nb of points per curve
    :return:
            inputs: points sample
            labels: points' label
    '''
    x_1 = np.linspace(0., 1.0, nb_points)
    x_2 = np.tanh(10 * (x_1 - 0.5) ** 2)
    inputs = np.zeros([2 * nb_points, 2], dtype=np.float32)
    labels = np.zeros([2 * nb_points], dtype=np.int32)
    inputs[:, 0] = np.append(x_1, x_1)
    inputs[:, 1] = np.append(x_2 - 0.9, x_2 - 0.1)
    labels[nb_points:] = 1

    return  inputs, labels

# generate spiral data
def spiral_data(nb_points = 100):
    inputs = np.zeros([2 * nb_points, 2])
    labels = np.zeros([2 * nb_points], dtype=np.int32)

    for i in range(nb_points):
        iper = i / nb_points
        r =  5 * iper + np.random.uniform(-0.1, 0.1)
        t = 1.25 * iper * 2. * np.pi + np.random.uniform(-0.1, 0.1)
        inputs[i] = [r * np.sin(t), r * np.cos(t)]
        r = 5 * iper + np.random.uniform(-0.1, 0.1)
        t = 1.25 * iper * 2. * np.pi + np.pi + np.random.uniform(-0.1, 0.1)
        inputs[i + nb_points] = [r * np.sin(t), r * np.cos(t)]
        labels[i + nb_points] = 1

    return inputs, labels