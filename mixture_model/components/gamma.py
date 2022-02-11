from scipy import special, optimize
from loguru import logger
import numpy as np

from ..base import MixtureModel, MixtureComponent


def incgamma(a, x):
    return special.gamma(a) * special.gammainc(a, x)


def cost_1d(x, w, t, alpha):
    N = x.shape[0]

    def cost(mu):
        log_likelihood = (
            N * alpha * np.log(alpha / mu)
            - N * np.log(incgamma(alpha, t * alpha / mu))
            - N * alpha / mu * np.sum(w * x, axis=0)
            + N * (alpha - 1) * np.sum(w * np.log(x), axis=0)
        )
        return -log_likelihood

    return cost


def cost_2d(x, w, t):
    N = x.shape[0]

    def cost(p):
        mu, alpha = p
        log_likelihood = (
            N * alpha * np.log(alpha / mu)
            - N * np.log(incgamma(alpha, t * alpha / mu))
            - N * alpha / mu * np.sum(w * x, axis=0)
            + N * (alpha - 1) * np.sum(w * np.log(x), axis=0)
        )
        return -log_likelihood

    return cost


class RightTruncatedUnivariateGamma(MixtureComponent):
    PARAMETERS = ["mu", "alpha", "truncation_value"]

    def __init__(self, *, mu=1, alpha=1, truncation_value=np.inf, fix_alpha=False):
        self.mu = mu
        self.alpha = alpha
        self.truncation_value = truncation_value
        self.dimensions = 1

        self.fix_alpha = fix_alpha

    @property
    def degrees_of_freedom(self):
        return 2

    def initialize_from_data(self, data):
        self.mu = np.mean(data, axis=0) + np.random.randn()

    def pdf(self, x, prior=1.0, extras=None):
        x = x.ravel()
        mu, alpha, t = self.mu, self.alpha, self.truncation_value
        value = prior * np.exp(
            alpha * np.log(alpha / mu)
            + (alpha - 1) * np.log(x)
            - x * alpha / mu
            - np.log(incgamma(alpha, t * alpha / mu))
        )
        value[x > t] = 0
        return value[:, None]

    def cdf(self, x, prior=1, extras=None):
        v = special.gammainc(self.alpha, x * self.alpha / self.mu) / special.gammainc(
            self.alpha, self.truncation_value * self.alpha / self.mu
        )
        v[x > self.truncation_value] = 1
        return v

    def update_parameters(self, data, weights, extras=None, iteration=None):
        x = data
        w = weights[:, None]
        w /= w.sum(axis=0)

        if not self.fix_alpha:
            logger.debug("Estimating mean and enl")
            inequality_constraints = {
                "type": "ineq",
                "fun": lambda x: np.array([x[0], x[1]]),  # mu > 0  # alpha > 0
            }

            initial_guess = (
                np.sum(w * x, axis=0),
                (w * x).std() ** 2 / (w * x).mean() ** 2,
            )
            result = optimize.minimize(
                cost_2d(x, w, self.truncation_value),
                initial_guess,
                method="SLSQP",
                constraints=[inequality_constraints],
            )
            self.mu, self.alpha = result.x
            logger.debug(f"Estimated mu: {self.mu}, alpha={self.alpha}")
        else:
            logger.debug("Estimating only mean")
            inequality_constraints = {
                "type": "ineq",
                "fun": lambda x: np.array([x[0]]),  # mu > 0
            }

            initial_guess = np.sum(w * x)
            result = optimize.minimize(
                cost_1d(x, w, self.truncation_value, self.alpha),
                initial_guess,
                method="SLSQP",
                constraints=[inequality_constraints],
            )
            self.mu = result.x
            logger.debug(f"Estimated mu: {self.mu}")

def plot_univariate():
    # return None
    import matplotlib.pyplot as plt

    likelihoods = []

    fig, ax = plt.subplots(2, 1)
    plt.show(block=False)
    plt.pause(0.5)

    def callback(**kwargs):
        iteration = kwargs["iteration"]
        max_iterations = kwargs["max_iterations"]
        has_converged = kwargs["has_converged"]

        data = kwargs["data"]
        model = kwargs["self"]
        relative_difference = kwargs["relative_difference"]

        ll = kwargs["log_likelihood"]
        likelihoods.append(ll)

        for a in ax:
            a.cla()

        ax[0].plot(likelihoods, "b.")
        ax[0].set_title(relative_difference)

        percentiles = np.percentile(data.ravel(), np.linspace(0, 100, 100))
        counts, bins = np.histogram(data.ravel(), percentiles, density=True)
        bin_centers = (bins[:-1] + bins[1:]) / 2.0
        ax[1].plot(bin_centers, counts, "g-")
        t = np.linspace(np.min(data), np.max(data), 200)
        for m, component in enumerate(model.components):
            ax[1].plot(t, component.pdf(t, prior=model.priors[m]), "k-")
        ax[1].plot(
            t,
            sum(
                component.pdf(t, prior=p)
                for (p, component) in zip(model.priors, model.components)
            ),
            "r--",
        )
        # plt.draw()
        ax[1].set_xlabel("Feature value")
        ax[1].set_ylabel("Density")
        ax[1].set_title(f"Iteration: {iteration}")

        if iteration == max_iterations - 1 or has_converged:
            plt.show(block=True)
        else:
            fig.canvas.draw_idle()
            plt.pause(0.01)
            plt.show(block=False)

    return callback
