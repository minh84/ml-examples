import numpy as np

# generate circle-data
def cicle_data(nb_points=50):
    inputs  = np.zeros([2*nb_points,2])
    outputs = np.zeros([2*nb_points], dtype=np.int32)
    for i in range(nb_points):
        r,t = np.random.uniform(0., 2.), np.random.uniform(0., 2.*np.pi)
        inputs[i] = [r*np.sin(t), r*np.cos(t)]
        r,t = np.random.uniform(3., 5.), np.random.uniform(0., 2.*np.pi)
        inputs[i+nb_points] = [r*np.sin(t), r*np.cos(t)]
        outputs[i+nb_points] = 1

    return inputs, outputs

# generate 2d-curve
def curve2d_data(nb_points=50):
    x_1 = np.linspace(0., 1.0, nb_points)
    x_2 = np.tanh(10 * (x_1 - 0.5) ** 2)
    inputs = np.zeros([2 * nb_points, 2], dtype=np.float32)
    outputs = np.zeros([2 * nb_points], dtype=np.int32)
    inputs[:, 0] = np.append(x_1, x_1)
    inputs[:, 1] = np.append(x_2 - 0.9, x_2 - 0.1)
    outputs[nb_points:] = 1

    return  inputs, outputs