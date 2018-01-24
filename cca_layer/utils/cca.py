
import numpy as np
from scipy.linalg import sqrtm


class CCA(object):
    """
    Cannonical correlation analysis
    """

    def __init__(self, r1=1e-3, r2=1e-3, rT=1e-3, method='svd'):
        """ Constructor """

        self.r1 = r1
        self.r2 = r2
        self.rT = rT
        self.method = method

        self.m1 = None
        self.m2 = None

        self.U = None
        self.V = None

    def fit(self, H1, H2, verbose=False):
        """ Compute projections into correlation space """

        # number of observations
        m = H1.shape[0]

        # compute mean vectors
        self.m1 = np.mean(H1, axis=0)
        self.m2 = np.mean(H2, axis=0)

        # center data
        H1bar = H1 - self.m1
        H2bar = H2 - self.m2

        # transpose to formulas in paper (columns are examples)
        H1bar = H1bar.T
        H2bar = H2bar.T

        # cross-covariance
        SigmaHat12 = (1.0 / (m - 1)) * np.dot(H1bar, H2bar.T)
        SigmaHat21 = SigmaHat12.T

        # covariance 1
        SigmaHat11 = (1.0 / (m - 1)) * np.dot(H1bar, H1bar.T)
        SigmaHat11 = SigmaHat11 + self.r1 * np.identity(SigmaHat11.shape[0])

        # covariance 2
        SigmaHat22 = (1.0 / (m - 1)) * np.dot(H2bar, H2bar.T)
        SigmaHat22 = SigmaHat22 + self.r2 * np.identity(SigmaHat22.shape[0])

        if self.method == 'theano-2':

            # compute T and correlation
            S11 = np.linalg.inv(np.linalg.cholesky(SigmaHat11))
            S22 = np.linalg.inv(np.linalg.cholesky(SigmaHat22))
            S22_inv = np.linalg.inv(SigmaHat22)
            S11_inv = np.linalg.inv(SigmaHat11)

            M1 = S11.dot(SigmaHat12).dot(S22_inv).dot(SigmaHat21).dot(S11.T)
            M2 = S22.dot(SigmaHat21).dot(S11_inv).dot(SigmaHat12).dot(S22.T)

            Values, E = np.linalg.eigh(M1)

            Coeffs = np.sqrt(Values[::-1])

            _, F = np.linalg.eigh(M2)

            E = E[:, ::-1]
            F = F[:, ::-1]

            # keep cca transformation matrix
            self.U = S11.T.dot(E)
            self.V = S22.T.dot(F)

            # workaround to flip axis of projection vector
            V1 = np.dot(H1bar.T, self.U)
            V2 = np.dot(H2bar.T, self.V)
            for d in xrange(H1bar.shape[0]):
                X = np.hstack([V1[:, d:d+1], V2[:, d:d+1]])
                X -= X.mean(axis=0)
                C = X.T.dot(X)
                c = C[0, 1] / (np.sqrt(C[0, 0]) * np.sqrt(C[1, 1]))
                self.V[:, d] *= np.sign(c)

        elif self.method == 'theano-3':
            import theano
            import theano.tensor as T

            H1_t = T.matrix('H1')
            H2_t = T.matrix('H2')

            # train set size
            m = H1_t.shape[0]

            # hidden representations
            H1bar_t = H1_t - T.mean(H1_t, axis=0)
            H2bar_t = H2_t - T.mean(H2_t, axis=0)

            # transpose to formulas in paper
            H1bar_t = H1bar_t.T
            H2bar_t = H2bar_t.T

            # cross-covariance
            SigmaHat12_t = (1.0 / (m - 1)) * T.dot(H1bar_t, H2bar_t.T)
            SigmaHat21_t = SigmaHat12_t.T

            # covariance 1
            SigmaHat11_t = (1.0 / (m - 1)) * T.dot(H1bar_t, H1bar_t.T)
            SigmaHat11_t = SigmaHat11_t + self.r1 * T.identity_like(SigmaHat11_t)

            # covariance 2
            SigmaHat22_t = (1.0 / (m - 1)) * T.dot(H2bar_t, H2bar_t.T)
            SigmaHat22_t = SigmaHat22_t + self.r2 * T.identity_like(SigmaHat22_t)

            # theano optimized version of paper
            S11c_t = T.slinalg.cholesky(SigmaHat11_t)
            S11ci_t = T.nlinalg.matrix_inverse(S11c_t)
            S11_inv_t = T.nlinalg.matrix_inverse(SigmaHat11_t)

            S22c_t = T.slinalg.cholesky(SigmaHat22_t)
            S22ci_t = T.nlinalg.matrix_inverse(S22c_t)
            S22_inv_t = T.nlinalg.matrix_inverse(SigmaHat22_t)

            # compute correlation (regularized)
            M1_t = S11ci_t.dot(SigmaHat12_t).dot(S22_inv_t).dot(SigmaHat21_t).dot(S11ci_t.T)
            M2_t = S22ci_t.dot(SigmaHat21_t).dot(S11_inv_t).dot(SigmaHat12_t).dot(S22ci_t.T)

            M1_t += self.rT * T.identity_like(M1_t)
            M2_t += self.rT * T.identity_like(M2_t)

            # compute eigen decomposition
            E1_t, E_t = T.nlinalg.eigh(M1_t)
            _, F_t = T.nlinalg.eigh(M2_t)

            # re-order output
            E1_t = E1_t[::-1]
            E_t = E_t[:, ::-1]
            F_t = F_t[:, ::-1]

            # compute projection matrices
            U_t = S11ci_t.T.dot(E_t)
            V_t = S22ci_t.T.dot(F_t)

            # project data
            lv1_cca_t = H1bar_t.T.dot(U_t)
            lv2_cca_t = H2bar_t.T.dot(V_t)

            # workaround to flip axis of projection vector
            def compute_corr(d, lv1_cca_t, lv2_cca_t):
                CX = lv1_cca_t[:, d].T.dot(lv2_cca_t[:, d])
                C1 = lv1_cca_t[:, d].T.dot(lv1_cca_t[:, d])
                C2 = lv2_cca_t[:, d].T.dot(lv2_cca_t[:, d])
                c = CX / (T.sqrt(C1) * T.sqrt(C2))
                return T.sgn(c)

            dims = T.arange(0, lv1_cca_t.shape[1])
            corrs_t, _ = theano.scan(fn=compute_corr, outputs_info=None, sequences=[dims], non_sequences=[lv1_cca_t, lv2_cca_t])

            V_t_fixed = V_t * corrs_t

            # compute correlation
            E1_t = T.clip(E1_t, 1e-7, 1.0)
            Coeffs_t = T.sqrt(E1_t)

            # compute actual output
            compute = theano.function([H1_t, H2_t], [U_t, V_t_fixed, Coeffs_t])
            self.U, self.V, Coeffs = compute(H1, H2)

        elif self.method == 'tuw':
            # like 'eigen', but computing M1 and M2 a bit more complicated

            S11 = np.linalg.inv(sqrtm(SigmaHat11))
            S22 = np.linalg.inv(sqrtm(SigmaHat22))
            S22_inv = np.linalg.inv(SigmaHat22)
            S11_inv = np.linalg.inv(SigmaHat11)

            M1 = S11.dot(SigmaHat12).dot(S22_inv).dot(SigmaHat21).dot(S11)
            M2 = S22.dot(SigmaHat21).dot(S11_inv).dot(SigmaHat12).dot(S22)

            Values, E = np.linalg.eigh(M1)
            Coeffs = np.sqrt(Values[::-1])

            _, F = np.linalg.eigh(M2)

            E = E[:, ::-1]
            F = F[:, ::-1]

            self.U = S11.dot(E)
            self.V = S22.dot(F)

            # flip signs of projections to match
            s = np.sign(self.U.T.dot(SigmaHat12).dot(self.V).diagonal())
            self.U *= s

        elif self.method == 'svd':

            S11 = np.linalg.inv(sqrtm(SigmaHat11))
            S22 = np.linalg.inv(sqrtm(SigmaHat22))

            Tnp = S11.dot(SigmaHat12).dot(S22)

            U, Values, V = np.linalg.svd(Tnp)

            Coeffs = Values

            self.U = S11.dot(U)
            self.V = S22.dot(V.T)

        elif self.method == 'svd-2':
            # like 'svd', but computes S11 and S22 via diagonalization

            d, A = np.linalg.eigh(SigmaHat11)
            S11 = (A * np.reciprocal(np.sqrt(d))).dot(A.T)  # = SigmaHat11^-.5
            d, A = np.linalg.eigh(SigmaHat22)
            S22 = (A * np.reciprocal(np.sqrt(d))).dot(A.T)  # = SigmaHat22^-.5

            Tnp = S11.dot(SigmaHat12).dot(S22)

            U, Values, V = np.linalg.svd(Tnp)

            Coeffs = Values

            self.U = S11.dot(U)
            self.V = S22.dot(V.T)

        elif self.method == 'eigen':
            # like 'svd', but computes U and V via two eigen decompositions

            S11 = np.linalg.inv(sqrtm(SigmaHat11))
            S22 = np.linalg.inv(sqrtm(SigmaHat22))

            Tnp = S11.dot(SigmaHat12).dot(S22)

            M1 = Tnp.dot(Tnp.T)
            M2 = Tnp.T.dot(Tnp)
            Values, E = np.linalg.eigh(M1)
            _, F = np.linalg.eigh(M2)
            E = E[:, ::-1]
            F = F[:, ::-1]
            Coeffs = np.sqrt(Values[::-1])

            self.U = S11.dot(E)
            self.V = S22.dot(F)

            # flip signs of projections to match
            # (needed because we do two decompositions as opposed to a SVD)
            s = np.sign(self.U.T.dot(SigmaHat12).dot(self.V).diagonal())
            self.U *= s

        elif self.method == 'eigen-2':
            # like 'eigen', but computes S11 and S22 via diagonalization
            d, A = np.linalg.eigh(SigmaHat11)
            S11 = (A * np.reciprocal(np.sqrt(d))).dot(A.T)  # = SigmaHat11^-.5
            d, A = np.linalg.eigh(SigmaHat22)
            S22 = (A * np.reciprocal(np.sqrt(d))).dot(A.T)  # = SigmaHat22^-.5

            Tnp = S11.dot(SigmaHat12).dot(S22)

            M1 = Tnp.dot(Tnp.T)
            M2 = Tnp.T.dot(Tnp)
            Values, E = np.linalg.eigh(M1)
            _, F = np.linalg.eigh(M2)
            E = E[:, ::-1]
            F = F[:, ::-1]
            Coeffs = np.sqrt(Values[::-1])

            self.U = S11.dot(E)
            self.V = S22.dot(F)

            # flip signs of projections to match
            # (needed because we do two decompositions as opposed to a SVD)
            s = np.sign(self.U.T.dot(SigmaHat12).dot(self.V).diagonal())
            self.U *= s

        elif self.method == 'eigen-3':
            # like 'eigen', but computes M1 and M2 via cholesky decomposition
            S11 = np.linalg.inv(np.linalg.cholesky(SigmaHat11))
            S22 = np.linalg.inv(np.linalg.cholesky(SigmaHat22))

            M1 = S11.dot(SigmaHat12).dot(S22.T).dot(S22).dot(SigmaHat21).dot(S11.T)
            M2 = S22.dot(SigmaHat21).dot(S11.T).dot(S11).dot(SigmaHat12).dot(S22.T)
            Values, E = np.linalg.eigh(M1)
            _, F = np.linalg.eigh(M2)
            E = E[:, ::-1]
            F = F[:, ::-1]
            Coeffs = np.sqrt(Values[::-1])

            self.U = S11.T.dot(E)
            self.V = S22.T.dot(F)

            # flip signs of projections to match
            # (needed because we do two decompositions as opposed to a SVD)
            s = np.sign(self.U.T.dot(SigmaHat12).dot(self.V).diagonal())
            self.U *= s

        elif self.method == 'eigen-3b':
            # like 'eigen', but computes M1 and M2 via cholesky decomposition
            S11 = np.linalg.inv(np.linalg.cholesky(SigmaHat11))
            S22 = np.linalg.inv(np.linalg.cholesky(SigmaHat22))

            Tnp = S11.dot(SigmaHat12).dot(S22.T)
            M1 = Tnp.dot(Tnp.T)
            M2 = Tnp.T.dot(Tnp)
            Values, E = np.linalg.eigh(M1)
            _, F = np.linalg.eigh(M2)
            E = E[:, ::-1]
            F = F[:, ::-1]
            Coeffs = np.sqrt(Values[::-1])

            self.U = S11.T.dot(E)
            self.V = S22.T.dot(F)

            # flip signs of projections to match
            # (needed because we do two decompositions as opposed to a SVD)
            s = np.sign(self.U.T.dot(SigmaHat12).dot(self.V).diagonal())
            self.U *= s

        elif self.method == 'eigen-4':
            # like 'eigen-3', but using a single eigenvalue decomposition only
            # see page 6 of Hardoon et al. (2004)
            # http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.702.5978&rep=rep1&type=pdf
            S11ci = np.linalg.inv(np.linalg.cholesky(SigmaHat11))
            S22i = np.linalg.inv(SigmaHat22)

            M1 = S11ci.dot(SigmaHat12).dot(S22i).dot(SigmaHat21).dot(S11ci.T)
            Values, E = np.linalg.eigh(M1)
            E = E[:, ::-1]
            Coeffs = np.sqrt(Values[::-1])

            self.U = S11ci.T.dot(E)
            self.V = S22i.dot(SigmaHat21).dot(self.U) / Coeffs

        elif self.method == 'eigen-2-theano':
            # like 'eigen-2', but in Theano
            import theano
            import theano.tensor as T
            # input variables
            H1_t = T.matrix('H1')
            H2_t = T.matrix('H2')
            # train set size
            m = H1_t.shape[0]
            # mean centering
            H1bar_t = H1_t - T.mean(H1_t, axis=0)
            H2bar_t = H2_t - T.mean(H2_t, axis=0)
            # cross-covariance
            SigmaHat12_t = (1.0 / (m - 1)) * T.dot(H1bar_t.T, H2bar_t)
            SigmaHat21_t = SigmaHat12_t.T
            # covariance 1
            SigmaHat11_t = (1.0 / (m - 1)) * T.dot(H1bar_t.T, H1bar_t)
            SigmaHat11_t = SigmaHat11_t + self.r1 * T.identity_like(SigmaHat11_t)
            # covariance 2
            SigmaHat22_t = (1.0 / (m - 1)) * T.dot(H2bar_t.T, H2bar_t)
            SigmaHat22_t = SigmaHat22_t + self.r2 * T.identity_like(SigmaHat22_t)

            d, A = T.nlinalg.eigh(SigmaHat11_t)
            S11 = (A * np.reciprocal(np.sqrt(d))).dot(A.T)  # = SigmaHat11^-.5
            d, A = T.nlinalg.eigh(SigmaHat22_t)
            S22 = (A * np.reciprocal(np.sqrt(d))).dot(A.T)  # = SigmaHat22^-.5

            Tnp = S11.dot(SigmaHat12_t).dot(S22)

            M1 = Tnp.dot(Tnp.T)
            M2 = Tnp.T.dot(Tnp)
            Values, E = T.nlinalg.eigh(M1)
            _, F = T.nlinalg.eigh(M2)
            E = E[:, ::-1]
            F = F[:, ::-1]
            Coeffs = T.sqrt(Values[::-1])

            U = S11.dot(E)
            V = S22.dot(F)

            # flip signs of projections to match
            # (needed because we do two decompositions as opposed to a SVD)
            s = T.sgn(U.T.dot(SigmaHat12_t).dot(V).diagonal())
            U *= s

            # compile function
            fn = theano.function([H1_t, H2_t], [U, V, Coeffs])
            self.U, self.V, Coeffs = fn(H1, H2)

        elif self.method == 'eigen-4-theano':
            # like 'eigen-4', but in Theano
            import theano
            import theano.tensor as T
            # input variables
            H1_t = T.matrix('H1')
            H2_t = T.matrix('H2')
            # train set size
            m = H1_t.shape[0]
            # mean centering
            H1bar_t = H1_t - T.mean(H1_t, axis=0)
            H2bar_t = H2_t - T.mean(H2_t, axis=0)
            # cross-covariance
            SigmaHat12_t = (1.0 / (m - 1)) * T.dot(H1bar_t.T, H2bar_t)
            SigmaHat21_t = SigmaHat12_t.T
            # covariance 1
            SigmaHat11_t = (1.0 / (m - 1)) * T.dot(H1bar_t.T, H1bar_t)
            SigmaHat11_t = SigmaHat11_t + self.r1 * T.identity_like(SigmaHat11_t)
            # covariance 2
            SigmaHat22_t = (1.0 / (m - 1)) * T.dot(H2bar_t.T, H2bar_t)
            SigmaHat22_t = SigmaHat22_t + self.r2 * T.identity_like(SigmaHat22_t)

            S11ci = T.nlinalg.matrix_inverse(T.slinalg.cholesky(SigmaHat11_t))
            S22i = T.nlinalg.matrix_inverse(SigmaHat22_t)

            M1 = S11ci.dot(SigmaHat12_t).dot(S22i).dot(SigmaHat21_t).dot(S11ci.T)
            Values, E = T.nlinalg.eigh(M1)
            E = E[:, ::-1]
            Coeffs = T.sqrt(Values[::-1])

            U = S11ci.T.dot(E)
            V = S22i.dot(SigmaHat21_t).dot(U) / Coeffs

            # compile function
            fn = theano.function([H1_t, H2_t], [U, V, Coeffs])
            self.U, self.V, Coeffs = fn(H1, H2)

        else:
            raise NotImplementedError("Selected method for CCA not implemented!")

        if verbose:
            print "\nCorrelation-Coeffs:  ", np.around(Coeffs, 3)
            print "Canonical-Correlation:", np.sum(Coeffs) / H1.shape[1]

        return Coeffs

    def transform(self, X):
        """ Project data into cca space """
        Xbar = X - self.m1
        return np.dot(Xbar, self.U)

    def transform_V1(self, X):
        """ Project data into cca space """
        return self.transform(X)

    def transform_V2(self, Y):
        """ Project data into cca space """
        Ybar = Y - self.m2
        return np.dot(Ybar, self.V)
