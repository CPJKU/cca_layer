
import theano.tensor as T
from theano.tensor.extra_ops import fill_diagonal


def get_contrastive_loss_kiros(weight, gamma, symmetric=False):
    """ Compile contrastive loss (Kiros et al. 2014) """

    def loss(lv1, lv2):
        """ Contrastive cosine distance optimization target """

        # compute image-sentence score matrix
        scores = T.dot(lv1, lv2.T)
        diagonal = scores.diagonal()

        # compare every diagonal score to scores in its column (i.e, all contrastive images for each sentence)
        cost_s = T.maximum(0, gamma - diagonal + scores)
        # compare every diagonal score to scores in its row (i.e, all contrastive sentences for each image)
        cost_im = T.maximum(0, gamma - diagonal.reshape((-1, 1)) + scores)

        # clear diagonals
        cost_s = fill_diagonal(cost_s, 0)
        cost_im = fill_diagonal(cost_im, 0)

        return cost_s.sum() + cost_im.sum()

    return loss


def get_contrastive_cos_loss(weight, gamma, symmetric=False):
    """ Compile contrastive loss (Kiros et al. 2014) """

    def loss(lv1, lv2):
        """ Contrastive cosine distance optimization target """

        n = lv1.shape[0]

        # direction 1
        D = lv1.dot(lv2.T)
        d = D.diagonal().reshape((-1, 1))

        M = T.identity_like(D)
        O = D[(M <= 0).nonzero()].reshape((n, n - 1))

        L = gamma - d
        L = T.repeat(L, n - 1, 1)
        L += O
        L = T.clip(L, 0, 1000)

        loss = L.mean()

        # direction 2
        if symmetric:
            D = lv2.dot(lv1.T)
            d = D.diagonal().reshape((-1, 1))

            M = T.identity_like(D)
            O = D[(M <= 0).nonzero()].reshape((n, n - 1))

            L = gamma - d
            L = T.repeat(L, n - 1, 1)
            L += O
            L = T.clip(L, 0, 1000)

            loss += L.mean()

        return weight * loss

    return loss


def get_contrastive_arccos_loss(weight, gamma):
    """ Compile contrastive loss (Kiros et al. 2014) """

    def loss(lv1, lv2):
        """ Contrastive cosine distance optimization target """

        # number of samples in batch
        n = lv1.shape[0]

        # compute cosine distance
        D = lv1.dot(lv2.T)

        # compute arcus cosinus -> converts similarity into distance
        D = T.arccos(D)

        # distance between matching pairs
        d = D.diagonal().reshape((-1, 1))

        # distance between non-matching pairs
        M = T.identity_like(D)
        O = D[(M <= 0).nonzero()].reshape((n, n - 1))

        # max margin hinge loss
        L = gamma + d
        L = T.repeat(L, n - 1, 1)
        L -= O
        L = T.clip(L, 0, 1000)

        # compute batch mean
        loss = L.mean()

        return weight * loss

    return loss


def get_cos2_distance_loss(WEIGHT):

    def loss(lv1, lv2):
        """ Squared cosine distance optimization target """

        D1 = (lv1 * lv2).sum(axis=-1)
        loss = T.mean(T.square(1.0 - D1))

        return (1.0 - WEIGHT) * loss

    return loss
