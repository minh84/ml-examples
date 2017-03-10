import numpy as np
import time


def sgd_np(f, initW, train_data, val_data, reg, epochs, learning_rate=1.0e-3, print_every=20):
    '''
    sgd_np implements SGD algorithm to minimize function f
        f: is a function with signature f(X, y, W, reg) => loss, grad
        initW: is initial weights
        train_data: is Dataset object supports function next_batch() => X_batch, y_batch used in train-step
        val_data: is Dataset object supports function next_batch() => X_batch, y_batch used in val-step
        reg: regularization lambda
        learning_rate: a hyperparameter to control update-step W:= W - learning_rate * dW
        print_every: log to console the loss & store it in loss_history to visualize it laters
    '''

    # downcast to float32
    W = initW.astype(np.float32)

    # get number of iteration
    nb_iters = train_data.get_nb_iters(epochs)
    loss_history = []

    start = time.time()
    for i in range(1, nb_iters + 1):
        X_batch, y_batch = train_data.next_batch()
        loss, grad = f(W, X_batch, y_batch, reg)
        loss_history.append(loss)

        it_per_second = i / (time.time() - start)
        # sys.stdout.write("\rProgress: {:>5.2f}% Speed (it/sec): {:>10.4f}".format(100 * i / nb_iters, it_per_second))

        # sgd update for minimize loss
        W -= learning_rate * grad

        # log current state
        if (i % print_every == 0):
            print('Iter {:>10d}/{:<10d} loss {:10.4f}'.format(i, nb_iters, loss))

        epoch_end, epoch = train_data.is_epoch_end(i)
        if (epoch_end) or (i == 1):
            # validation it here
            if val_data is not None:
                X_val, y_val = val_data.next_batch()
                scores = f(W, X_val, None, reg)
                acc = np.mean(np.argmax(scores, axis=1) == y_val)
                print('\nEpoch {:>3d}/{:<3d} val_acc = {:5.2f}%'.format(epoch, epochs, 100 * acc))

    print ('\nTrain time: {:<10.2f} seconds'.format(time.time() - start))
    return W, loss_history