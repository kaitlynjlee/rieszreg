"""The two estimands in the simulation, the ATE and the ATT.

For each estimand this file holds what the simulation needs:

    riesz_estimand     the rieszreg estimand that the Riesz learners fit
    true_representer   alpha_0(A, X) under a given DGP
    true_value         psi_0 under a given DGP, by Monte Carlo over X, with
                       the Monte Carlo SE of that approximation
    one_step           the one-step estimate of psi and its standard error,
                       from cross-fitted mu_hat and alpha_hat

The estimators never fit anything. They only read nuisance predictions, so
adding an estimator refits no learner.

Notation: pi(X) = P(A = 1 | X), mu(a, X) = E[Y | A = a, X], and
tau(X) = mu(1, X) - mu(0, X).
"""

import numpy as np
import rieszreg


class AverageTreatmentEffect:
    """psi = E[tau(X)].

    Riesz representer: alpha(A, X) = A / pi(X) - (1 - A) / (1 - pi(X)).
    One-step estimator: the sample mean of the estimated influence function
        phi(O) = tau_hat(X) + alpha_hat(A, X) (Y - mu_hat(A, X)),
    with standard error sd(phi) / sqrt(n).
    """

    name = "ATE"

    def riesz_estimand(self, covariates):
        return rieszreg.ATE(treatment="a", covariates=tuple(covariates))

    def true_representer(self, dgp, A, X):
        pi = dgp.propensity(X)
        return A / pi - (1 - A) / (1 - pi)

    def true_value(self, dgp, rng, n_mc):
        X = dgp.draw_covariates(n_mc, rng)
        tau = dgp.outcome_regression(1.0, X) - dgp.outcome_regression(0.0, X)
        return float(tau.mean()), float(tau.std() / np.sqrt(n_mc))

    def one_step(self, A, Y, mu, mu1, mu0, alpha):
        phi = mu1 - mu0 + alpha * (Y - mu)
        return phi.mean(), phi.std(ddof=1) / np.sqrt(len(Y))


class AverageTreatmentEffectOnTreated:
    """psi = E[tau(X) | A = 1] = psi_star / P(A = 1), with psi_star = E[A tau(X)].

    The Riesz learners fit the representer of psi_star (rieszreg's ATT),
        alpha(A, X) = A - (1 - A) pi(X) / (1 - pi(X)).
    One-step estimator, with p_hat the sample mean of A:
        psi_hat = mean[A tau_hat(X) + alpha_hat (Y - mu_hat)] / p_hat
        phi(O)  = [A (tau_hat(X) - psi_hat) + alpha_hat (Y - mu_hat)] / p_hat,
    with standard error sd(phi) / sqrt(n).
    """

    name = "ATT"

    def riesz_estimand(self, covariates):
        return rieszreg.ATT(treatment="a", covariates=tuple(covariates))

    def true_representer(self, dgp, A, X):
        pi = dgp.propensity(X)
        return A - (1 - A) * pi / (1 - pi)

    def true_value(self, dgp, rng, n_mc):
        # Weight tau(X) by pi(X) instead of drawing A: same expectation, less
        # Monte Carlo variance. The SE is the delta-method SE of the ratio.
        X = dgp.draw_covariates(n_mc, rng)
        pi = dgp.propensity(X)
        tau = dgp.outcome_regression(1.0, X) - dgp.outcome_regression(0.0, X)
        psi = np.sum(pi * tau) / np.sum(pi)
        return float(psi), float(np.std(pi * (tau - psi)) / (pi.mean() * np.sqrt(n_mc)))

    def one_step(self, A, Y, mu, mu1, mu0, alpha):
        p = A.mean()
        est = np.mean(A * (mu1 - mu0) + alpha * (Y - mu)) / p
        phi = (A * (mu1 - mu0 - est) + alpha * (Y - mu)) / p
        return est, phi.std(ddof=1) / np.sqrt(len(Y))


ESTIMANDS = {e.name: e for e in (AverageTreatmentEffect(), AverageTreatmentEffectOnTreated())}
