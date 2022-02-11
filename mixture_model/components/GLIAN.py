import scipy.stats
import numpy as np
import numpy.polynomial.polynomial as poly
from loguru import logger

from ..base import MixtureComponent


def create_positive_semidefinate_matrix(N, scale=1):
    c = scale * np.random.randn(N, N)
    c = c @ c.T
    return c


def simulate_multivariate_gaussian(number_of_samples, mu, cov):
    return np.random.multivariate_normal(
        mu, cov, size=number_of_samples, check_valid="raise"
    )

class GLIA(MixtureComponent):
    PARAMETERS = ["dimensions", "mu", "beta", "cov"]

    def __init__(self, dimensions=None, mu=None, beta=None, cov=None):
        if dimensions is None:
            raise ValueError("Need to know the number of dimensions")
        self.dimensions = dimensions
        self.mu = np.array(mu)
        self.beta = np.array(beta)
        self.cov = np.array(cov)

    @property
    def degrees_of_freedom(self):
        return 3

    def initialize_from_data(self, data, extras=None):
        mean_value = np.mean(data, axis=0)
        self.beta = np.zeros(data.shape[1])
        self.cov = np.atleast_2d(np.cov(data - mean_value, rowvar=False))
        self.mu = np.random.multivariate_normal(mean_value, self.cov)

    def noisefunc(self, extras):	
        #  NB: assume mu and beta are flattened size (d). Returns yy size [Nia, dd]
        #  NB: extras should contain: IA, NF, azNF, SSM, NLS
        Nia, dd = extras["NF"].shape
        ns = np.zeros((Nia, dd), dtype=np.float32)
        if (extras["Nsw"] > 1) and not np.allclose(extras["NLS"], 1.0):
            for sw in range(extras["Nsw"]):
                for di in range(dd):
                    ns[extras["SSM"]==sw+1, di] = extras["NLS"][sw] * extras["NF"][extras["SSM"]==sw+1, di] * extras["azNF"][extras["SSM"]==sw+1, di]
        else: # one sub-swath or all nlvls values ones.
            for di in range(dd):
                ns[:, di] = extras["NLS"][0]*extras["NF"][:, di]*extras["azNF"][:, di]
                
        return (ns)

    def varmeanfunc(self, extras):	
        #  NB: assume mu and beta are flattened size (d). Returns yy size [Nia, dd]
        #  NB: nlvls will override NLS in model. Used for optimization stage.
        Nia = extras["IA"].flatten().shape[0]
        dd = self.cov.shape[0]
        yy = np.tile(np.reshape(self.mu, (1, dd)), (Nia, 1))
        yy += np.tile(np.reshape(self.beta, (1, dd)), (Nia, 1)) * np.tile(extras["IA"].reshape(Nia, 1), (1, dd))
        # NOW DO NOISE-FLOOR STUFF.
        yy = 10.0**(yy/10.0) # in intensity/power domain
        yy += self.noisefunc(extras)
        yy = 10.0*np.log10(yy) # back in decibels
    
        return (yy)

    def pdf(self, x, prior=1, extras=None):
        devs = x - self.varmeanfunc(extras)
        distribution = scipy.stats.multivariate_normal(
            mean=np.zeros(self.cov.shape[0]), cov=self.cov, allow_singular=False
        )
        value = np.exp(distribution.logpdf(devs) + np.log(prior))
        return value

    def cdf(self, x, prior=1, extras=None):
        raise ValueError("CDF not sensible for  bi-variate GLIA")
        return (0.0)
        # distribution = scipy.stats.multivariate_normal(
        #    mean=self.mu, cov=self.cov, allow_singular=False
        # )
        # value = np.exp(distribution.logcdf(x) + np.log(prior))
        # return value

    def update_parameters(self, data, weights, extras=None, iteration=None):
        dd = self.cov.shape[0]
        wz = weights
        wzs = wz.sum(axis=0)
        wz /= wzs
        xd = 10.0*np.log10(10.0**(data/10.0) - self.noisefunc(extras))
        bad = np.count_nonzero(np.logical_not(np.isfinite(xd)))
        xd[np.logical_not(np.isfinite(xd))] = -50.0 - 10.0*np.random.rand((bad))
        for di in range(dd):
            # vld = np.isfinite(wz) * np.logical_not(wz == 0) * np.isfinite(xd)
            pcoefs = poly.polyfit(extras["IA"], xd[:, di], 1, w=wz, rcond=(np.nansum(wz > 0)*np.finfo(np.float).eps))
            self.mu[di] = pcoefs[0]
            self.beta[di] = pcoefs[1]
        devs = data - self.varmeanfunc(extras)
        self.cov = np.cov(devs, aweights=wz.ravel(), rowvar=False)
#         if wzs < dd:
#             logger.info("Empty class: Expanding cov.") # to try to find another area of data samples.
#             self.mu = np.mean(data, axis=0)
#             self.beta = np.zeros((dd))
#             self.cov = np.cov(devs, rowvar=False) # without weights, whole dataset.
#         else:
#             self.cov = np.cov(devs, aweights=wz.ravel(), rowvar=False)


def plot_univariate(number_of_bins=31):
    """Callback function for interactive plotting during training."""

    # It might seem weird that matplotlib is imported inside the function, but the idea is that
    # matplotlib is a large dependency, which can often be useful during development, but may not be necessary
    # for the analysis itself.
    import matplotlib.pyplot as plt

    log_likelihoods = []

    fig, ax = plt.subplots(2, 1)
    h_likelihood, = ax[0].plot([], [], "b*")
    h_histogram, = ax[1].plot([], [], "b-")

    component_plots = []

    plt.pause(0.5)
    plt.show(block=False)

    def callback(**kwargs):
        iteration = kwargs["iteration"]
        max_iterations = kwargs["max_iterations"]

        data = kwargs["data"]
        model = kwargs["self"]
        relative_difference = kwargs["relative_difference"]

        for h in component_plots:
            h.remove()
        component_plots.clear()

        ll = kwargs["log_likelihood"]
        log_likelihoods.append(ll)
        h_likelihood.set_data(np.arange(len(log_likelihoods)), log_likelihoods)
        # ax[0].set_title(relative_difference)

        percentiles = np.percentile(data.ravel(), np.linspace(0, 100, number_of_bins))
        counts, bins = np.histogram(data.ravel(), percentiles, density=True)
        bin_centers = (bins[:-1] + bins[1:]) / 2.0
        # ax[1].plot(bin_centers, counts, "g-")
        h_histogram.set_data(bin_centers, counts)
        t = np.linspace(np.min(data), np.max(data), 200)
        for m, component in enumerate(model.components):
            mu = component.mu
            cov = np.sqrt(float(component.cov))
            h, = ax[1].plot(
                t,
                component.pdf(t, prior=model.priors[m]),
                "k-",
                label=f"$\\pi_{m}$={model.priors[m]}, $\\mu_{m}$={mu}, $\\cov_{m}$={cov}",
            )
            component_plots.append(h)
            h = ax[1].vlines(mu, 0, counts.max())
            component_plots.append(h)
        h, = ax[1].plot(
            t,
            sum(
                p * component.pdf(t)
                for (p, component) in zip(model.priors, model.components)
            ),
            "r--",
        )
        component_plots.append(h)
        ax[0].set_ylabel("Log-likelihood")
        ax[1].set_xlabel("Feature value")
        ax[1].set_ylabel("Density")
        ax[1].set_title(f"Iteration: {iteration + 1} / {max_iterations}")

        fig.canvas.draw_idle()
        for a in ax:
            a.relim()
            a.autoscale_view()
        if iteration == max_iterations - 1:
            plt.waitforbuttonpress()
        else:
            plt.pause(0.1)

    return callback


def plot_multivariate():
    from scipy.stats import multivariate_normal
    import matplotlib.pyplot as plt

    likelihoods = []

    fig, ax = plt.subplots(2, 1)
    h_likelihood, = ax[0].plot([], [], "b*")
    h_histogram, = ax[1].plot([], [], "b-")

    component_plots = []

    plt.show(block=False)
    plt.pause(0.5)

    def callback(**kwargs):
        iteration = kwargs["iteration"]
        max_iterations = kwargs["max_iterations"]

        data = kwargs["data"]
        model = kwargs["self"]
        relative_difference = kwargs["relative_difference"]
        has_converged = kwargs["has_converged"]

        ll = kwargs["log_likelihood"]
        likelihoods.append(ll)
        h_likelihood.set_data(np.arange(len(likelihoods)), likelihoods)

        for h in component_plots:
            h.remove()
        component_plots.clear()

        # ax[0].plot(likelihoods, "b.")
        # ax[0].set_title(relative_difference)

        _, _, _, h_histogram = ax[1].hist2d(data[:, 0], data[:, 1], 50, cmin=1)

        vmin, vmax = data.min(), data.max()
        vmin = vmin - 0.5 * (vmax - vmin)
        vmax = vmax + 0.5 * (vmax - vmin)

        x, y = np.meshgrid(
            # np.linspace(data[:, 0].min(), data[:, 0].max(), 30),
            # np.linspace(data[:, 1].min(), data[:, 1].max(), 30),
            np.linspace(vmin, vmax, 100),
            np.linspace(vmin, vmax, 100),
        )

        for m, component in enumerate(model.components):
            density = multivariate_normal(mean=component.mu, cov=component.cov)
            z = (
                    density.pdf(np.vstack([x.ravel(), y.ravel()]).T).reshape(x.shape)
                    * model.priors[m]
            )
            h = ax[1].contour(x, y, z, 20, cmap="RdGy")
            component_plots.extend(h.collections)
        ax[1].set_aspect("equal", adjustable="box")

        fig.canvas.draw_idle()
        for a in ax:
            a.relim()
            a.autoscale_view()
        if has_converged or iteration == max_iterations - 1:
            plt.waitforbuttonpress()
        else:
            plt.pause(0.1)

    return callback
