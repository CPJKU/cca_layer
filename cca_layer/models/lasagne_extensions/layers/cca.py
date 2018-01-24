
import numpy as np

import theano
import theano.tensor as T
from lasagne.layers.base import MergeLayer, Layer
from lasagne import init
from lasagne.init import floatX


__all__ = ["CCALayer", "LengthNormLayer"]


class NormLayer(Layer):
    """
    Normalize network output to length 1.0
    """
    def __init__(self, incoming, norm_value=None, **kwargs):
        super(NormLayer, self).__init__(incoming, **kwargs)
        self.norm_value = np.float32(norm_value)

    def get_output_shape_for(self, input_shape):
        return input_shape

    def get_output_for(self, incoming, **kwargs):
        return incoming / self.norm_value


class LengthNormLayer(Layer):
    """
    Normalize network output to length 1.0
    """
    def __init__(self, incoming, **kwargs):
        super(LengthNormLayer, self).__init__(incoming, **kwargs)

    def get_output_shape_for(self, input_shape):
        return input_shape

    def get_output_for(self, incoming, **kwargs):
        return incoming / incoming.norm(2, axis=1).reshape((incoming.shape[0], 1))


class CCALayer(MergeLayer):
    """
    Canonical Correlation Layer
    """

    def __init__(self, incomings, r1, r2, rT, U=init.Constant(0), V=init.Constant(0),
                 S12=init.Constant(0), S11=init.Constant(0), S22=init.Constant(0),
                 mean1=init.Constant(0), mean2=init.Constant(0),
                 alpha=1.0, wl=0.0, normalized=False, **kwargs):
        super(CCALayer, self).__init__(incomings, **kwargs)

        # dimensionality of hidden space
        self.num_units = incomings[0].output_shape[1]

        self.r1 = r1
        self.r2 = r2
        self.rT = rT

        self.alpha = floatX(alpha)
        self.wl = floatX(wl)
        self.normalized = normalized

        self.loss = T.constant(0.0)
        self.corr = T.constant(0.0)

        num_inputs = int(np.prod(self.input_shapes[0][1:]))
        self.U = self.add_param(U, (num_inputs, self.num_units), name="U", trainable=False, regularizable=False)
        self.V = self.add_param(V, (num_inputs, self.num_units), name="V", trainable=False, regularizable=False)

        self.mean1 = self.add_param(mean1, (self.num_units,), name="mean1", trainable=False, regularizable=False)
        self.mean2 = self.add_param(mean2, (self.num_units,), name="mean2", trainable=False, regularizable=False)

        self.S12 = self.add_param(S12, (num_inputs, self.num_units), name="S12", trainable=False, regularizable=False)
        self.S11 = self.add_param(S11, (num_inputs, self.num_units), name="S11", trainable=False, regularizable=False)
        self.S22 = self.add_param(S22, (num_inputs, self.num_units), name="S22", trainable=False, regularizable=False)

    def get_output_shape_for(self, input_shapes):
        return input_shapes[0][0], 2 * self.num_units

    def get_output_for(self, inputs, deterministic=False, **kwargs):

        # extract inputs
        H1, H2 = inputs

        # train set size
        m = H1.shape[0].astype(theano.config.floatX)

        # running average projection matrix update
        if not deterministic:

            # compute batch mean
            mean1 = T.mean(H1, axis=0)
            mean2 = T.mean(H2, axis=0)

            # running average updates of means
            mean1 = (floatX(1.0 - self.alpha) * self.mean1 + self.alpha * mean1)
            running_mean1 = theano.clone(self.mean1, share_inputs=False)
            running_mean1.default_update = mean1
            mean1 += 0 * running_mean1

            mean2 = (floatX(1.0 - self.alpha) * self.mean2 + self.alpha * mean2)
            running_mean2 = theano.clone(self.mean2, share_inputs=False)
            running_mean2.default_update = mean2
            mean2 += 0 * running_mean2

            # hidden representations
            H1bar = H1 - mean1
            H2bar = H2 - mean2

            # transpose to formulas in paper
            H1bar = H1bar.T
            H2bar = H2bar.T

            # cross-covariance
            S12 = (1.0 / (m - 1)) * T.dot(H1bar, H2bar.T)

            # covariance 1
            S11 = (1.0 / (m - 1)) * T.dot(H1bar, H1bar.T)
            S11 = S11 + self.r1 * T.identity_like(S11)

            # covariance 2
            S22 = (1.0 / (m - 1)) * T.dot(H2bar, H2bar.T)
            S22 = S22 + self.r2 * T.identity_like(S22)

            # running average updates of statistics
            S12 = (floatX(1.0 - self.alpha) * self.S12 + self.alpha * S12)
            running_S12 = theano.clone(self.S12, share_inputs=False)
            running_S12.default_update = S12
            S12 += 0 * running_S12

            S11 = (floatX(1.0 - self.alpha) * self.S11 + self.alpha * S11)
            running_S11 = theano.clone(self.S11, share_inputs=False)
            running_S11.default_update = S11
            S11 += 0 * running_S11

            S22 = (floatX(1.0 - self.alpha) * self.S22 + self.alpha * S22)
            running_S22 = theano.clone(self.S22, share_inputs=False)
            running_S22.default_update = S22
            S22 += 0 * running_S22

            # theano-compatible formulation of paper
            d, A = T.nlinalg.eigh(S11)
            S11si = (A * np.reciprocal(np.sqrt(d))).dot(A.T)  # = S11^-.5
            d, A = T.nlinalg.eigh(S22)
            S22si = (A * np.reciprocal(np.sqrt(d))).dot(A.T)  # = S22^-.5

            # compute TT' and T'T (regularized)
            Tnp = S11si.dot(S12).dot(S22si)
            M1 = Tnp.dot(Tnp.T)
            M2 = Tnp.T.dot(Tnp)
            M1 += self.rT * T.identity_like(M1)
            M2 += self.rT * T.identity_like(M2)

            # compute eigen decomposition
            E1, E = T.nlinalg.eigh(M1)
            _, F = T.nlinalg.eigh(M2)

            # maximize correlation
            E1 = T.clip(E1, 1e-7, 1.0)
            E1 = T.sqrt(E1)
            self.loss = -T.mean(E1) * self.wl
            self.corr = E1

            # compute projection matrices
            U = S11si.dot(E)
            V = S22si.dot(F)

            # flip signs of projections to match
            # (needed because we do two decompositions as opposed to a SVD)
            s = T.sgn(U.T.dot(S12).dot(V).diagonal())
            U *= s

            # update of projection matrices
            running_U = theano.clone(self.U, share_inputs=False)
            running_U.default_update = U
            U += floatX(0) * running_U

            running_V = theano.clone(self.V, share_inputs=False)
            running_V.default_update = V
            V += floatX(0) * running_V

        # use projections of layer
        else:

            # hidden representations
            H1bar = H1 - self.mean1
            H2bar = H2 - self.mean2

            # transpose to formulas in paper
            H1bar = H1bar.T
            H2bar = H2bar.T

            U, V = self.U, self.V

        # re-project data
        lv1_cca = H1bar.T.dot(U)
        lv2_cca_fixed = H2bar.T.dot(V)

        output = T.horizontal_stack(lv1_cca, lv2_cca_fixed)

        return output

    def get_loss(self):
        return self.loss

    def get_corr(self):
        return self.corr


class LearnedCCALayer(MergeLayer):
    """
    Canonical Correlation Layer
    """

    def __init__(self, incomings, U=init.Constant(0), V=init.Constant(0), alpha=1.0,
                 mean1=init.Constant(0), mean2=init.Constant(0), **kwargs):
        super(LearnedCCALayer, self).__init__(incomings, **kwargs)

        # dimensionality of hidden space
        self.num_units = incomings[0].output_shape[1]
        self.alpha = floatX(alpha)

        self.loss = T.constant(0.0)
        self.corr = T.constant(0.0)
        self.r1 = self.r2 = self.rT = 1e-3

        num_inputs = int(np.prod(self.input_shapes[0][1:]))
        self.U = self.add_param(U, (num_inputs, self.num_units), name="U", trainable=True, regularizable=True)
        self.V = self.add_param(V, (num_inputs, self.num_units), name="V", trainable=True, regularizable=True)

        self.mean1 = self.add_param(mean1, (self.num_units,), name="mean1", trainable=False, regularizable=False)
        self.mean2 = self.add_param(mean2, (self.num_units,), name="mean2", trainable=False, regularizable=False)

    def get_output_shape_for(self, input_shapes):
        return input_shapes[0][0], 2 * self.num_units

    def get_output_for(self, inputs, deterministic=False, **kwargs):

        # extract inputs
        H1, H2 = inputs

        # train set size
        m = H1.shape[0].astype(theano.config.floatX)

        # running average projection matrix update
        if not deterministic:

            # compute batch mean
            mean1 = T.mean(H1, axis=0)
            mean2 = T.mean(H2, axis=0)

            # running average updates of means
            mean1 = (floatX(1.0 - self.alpha) * self.mean1 + self.alpha * mean1)
            running_mean1 = theano.clone(self.mean1, share_inputs=False)
            running_mean1.default_update = mean1
            mean1 += 0 * running_mean1

            mean2 = (floatX(1.0 - self.alpha) * self.mean2 + self.alpha * mean2)
            running_mean2 = theano.clone(self.mean2, share_inputs=False)
            running_mean2.default_update = mean2
            mean2 += 0 * running_mean2

            # hidden representations
            H1bar = H1 - mean1
            H2bar = H2 - mean2

            # transpose to correlation format
            H1bar = H1bar.T
            H2bar = H2bar.T

            # cross-covariance
            S12 = (1.0 / (m - 1)) * T.dot(H1bar, H2bar.T)

            # covariance 1
            S11 = (1.0 / (m - 1)) * T.dot(H1bar, H1bar.T)
            S11 = S11 + self.r1 * T.identity_like(S11)

            # covariance 2
            S22 = (1.0 / (m - 1)) * T.dot(H2bar, H2bar.T)
            S22 = S22 + self.r2 * T.identity_like(S22)

            # theano-compatible formulation of paper
            d, A = T.nlinalg.eigh(S11)
            S11si = (A * np.reciprocal(np.sqrt(d))).dot(A.T)  # = S11^-.5
            d, A = T.nlinalg.eigh(S22)
            S22si = (A * np.reciprocal(np.sqrt(d))).dot(A.T)  # = S22^-.5

            # compute TT' and T'T (regularized)
            Tnp = S11si.dot(S12).dot(S22si)
            M1 = Tnp.dot(Tnp.T)
            M2 = Tnp.T.dot(Tnp)
            M1 += self.rT * T.identity_like(M1)
            M2 += self.rT * T.identity_like(M2)

            # compute eigen decomposition
            E1, E = T.nlinalg.eigh(M1)
            _, F = T.nlinalg.eigh(M2)

            # compute correlation
            E1 = T.clip(E1, 1e-7, 1.0)
            E1 = T.sqrt(E1)
            self.corr = E1

            # transpose back to network format
            H1bar = H1bar.T
            H2bar = H2bar.T

        # use means of layer
        else:

            # hidden representations
            H1bar = H1 - self.mean1
            H2bar = H2 - self.mean2

        # re-project data
        lv1_cca = H1bar.dot(self.U)
        lv2_cca = H2bar.dot(self.V)

        output = T.horizontal_stack(lv1_cca, lv2_cca)

        return output


    def get_loss(self):
        return self.loss

    def get_corr(self):
        return self.corr