from abc import ABC, abstractmethod
from loguru import logger
import numpy as np
import pathlib
import toml
import scipy

class MixtureModel:
    def __init__(self, components, priors=None, extras=None):
        number_of_classes = len(components)
        if number_of_classes == 0:
            raise ValueError("Model needs to be created with at least one component.")

        if not all(isinstance(c, MixtureComponent) for c in components):
            raise ValueError(
                f"Expected a list of mixture components, but got {components}"
            )

        if not all(c.dimensions == components[0].dimensions for c in components):
            raise ValueError(
                (
                    "Expected all components to work with same data dimensions, "
                    f"but got dimensions {[c.dimensions for c in components]}."
                )
            )
        self.components = np.array(components)

        if priors is None:
            priors = np.random.random(number_of_classes)
        self.priors = np.array(priors)


        logger.info(
            f"Created mixture model: components={self.components} prior={self.priors}"
        )

    def cdf(self, x):
        return np.sum(
            component.cdf(x, prior)
            for (prior, component) in zip(self.priors, self.components)
        )

    def fit(
        self,
        data,
        *,
        max_iterations=50,
        extras=None,
        callback=None,
        tolerance=1e-8,
        reinitialize=True,
    ):
        data = np.array(data)
        if not len(data.shape) == 2:
            raise ValueError((
                f"Expected two-dimensional data matrix with dimension "
                f"[number_of_samples, number_of_features] but got {data.shape}"
            ))

        if reinitialize:
            logger.warning('Reinitializing model')
            self.initialize_from_data(data)

        number_of_samples, number_of_features = data.shape
        number_of_classes = len(self.priors)
        logger.info((
            f"Fitting {number_of_samples} x {number_of_features} data matrix "
            f"into {number_of_classes} classes"
        ))

        last_log_likelihood = -np.inf

        for iteration in range(max_iterations):
            logger.debug(f"Iteration {iteration} / {max_iterations}")

            weights, normalization_factor = self.e_step(data, extras)
            self.m_step(data, weights, extras, iteration=iteration)

            log_likelihood = np.sum(
                np.log(np.sum(self.likelihood(data, extras), axis=1))
            )
            logger.debug(f"Log-likelihood: {last_log_likelihood}  --> {log_likelihood}")

            relative_difference = abs(
                (last_log_likelihood - log_likelihood) / log_likelihood
            )

            has_converged = relative_difference < tolerance
            if callback:
                logger.debug(f"Running callback for iteration {iteration}")
                callback(**locals())

            if has_converged:
                logger.info("Converged")
                return True

            last_log_likelihood = log_likelihood
        else:
            # Iteration finished without encountering a break, meaning it did not converge.
            logger.warning("Did not converge")
            return False

# ============================================================================
# Fit with adaptive noise-floor, per sub-swath levels.

    def optimFuncAN(self, xv, dat, wgts, exts):
        # will calculate a total squared error. Assume same NF for each dim.
        #logger.debug("dat"+str(dat.shape)+"  wgts"+str(wgts.shape))
        val = 0.0
        for mm in range(len(self.components)):
           # calc. devs and TSE with current test vec. xv for NFLs.
            devs2 = np.nansum((dat - self.components[mm].varmeanfunc(exts, xv))**2, axis=1)
            #logger.debug("min="+str(np.nanmin(devs))+" max="+str(np.nanmax(devs)))
            val += np.nansum(devs2[np.argmax(wgts, axis=1)==mm])
        #logger.debug("val " + str(val) + "    xv "+str(xv))
        return (val)


    def optimFuncAN1(self, xv, dat, wgts, exts):
        # will calculate a total squared error. Assume same NF for each dim.
        #logger.debug("dat"+str(dat.shape)+"  wgts"+str(wgts.shape))
        val = 0.0
        for mm in range(len(self.components)):
           # calc. devs and TSE with current test vec. xv for NFLs.
            devs = dat - self.components[mm].varmeanfunc(exts, xv)
            #logger.debug("min="+str(np.nanmin(devs))+" max="+str(np.nanmax(devs)))
            val += np.nansum(np.nansum((devs**2), axis=1).flatten()*wgts[:, mm].flatten())
        #logger.debug("val " + str(val) + "    xv "+str(xv))
        return (val)


    def annealAN(self, strt, dat, wgts, exts, bounds):
        newNLS = strt.copy() # np.ones((exts["Nws"]))
        scrs = np.zeros((3))
        scrs[0] = self.optimFuncAN(newNLS, dat, wgts, exts)
        for ni in range(exts["Nsw"]):
            for delta in (0.5, 0.1, 0.01): # , 0.001
                for mi in range(5):  # limit to a few moves only.
                    cplus = np.fmin(bounds[ni][1], newNLS[ni] + delta)
                    cminus = np.fmax(bounds[ni][0], newNLS[ni] - delta)
                    cvec = np.stack((newNLS[ni], cplus, cminus))
                    newNLS[ni] = cplus
                    scrs[1] = self.optimFuncAN(newNLS, dat, wgts, exts)
                    newNLS[ni] = cminus
                    scrs[2] = self.optimFuncAN(newNLS, dat, wgts, exts)
                    newNLS[ni] = cvec[0] # put back current one.
                    indx = np.argmin(scrs)
                    if indx > 0:
                        newNLS[ni] = cvec[indx]
                        scrs[0] = scrs[indx]
                    else:
                        break    # didn't move.
        return (newNLS)



    def fitAN(
        self,
        data,
        *,
        adaptive_noise=False,
        max_iterations=50,
        extras=None,
        callback=None,
        tolerance=1e-8,
        reinitialize=True,
    ):
        data = np.array(data)
        if not len(data.shape) == 2:
            raise ValueError((
                f"Expected two-dimensional data matrix with dimension "
                f"[number_of_samples, number_of_features] but got {data.shape}"
            ))

        if reinitialize:
            logger.warning('Reinitializing model')
            self.initialize_from_data(data)

        number_of_samples, number_of_features = data.shape
        number_of_classes = len(self.priors)
        logger.info((
            f"Fitting {number_of_samples} x {number_of_features} data matrix "
            f"into {number_of_classes} classes"
        ))

        last_log_likelihood = -np.inf

        for iteration in range(max_iterations):
            logger.info(f"Iteration {iteration} / {max_iterations}")

            weights, normalization_factor = self.e_step(data, extras)
            self.m_step(data, weights, extras, iteration=iteration)

            if adaptive_noise:
                #  now adjust noise-floor levels.
                boundsvec = (extras["Nsw"]*((0.75, 1.5),))
                #extras["NLS"] = scipy.optimize.minimize(self.optimFuncAN, extras["NLS"], args=(data, weights, extras), bounds=boundsvec, method='trust-constr').x
                extras["NLS"] = self.annealAN(extras["NLS"], data, weights, extras, boundsvec)
                logger.debug("Adaptive Noise Adjustment: " + str(extras["NLS"]))


            log_likelihood = np.sum(
                np.log(np.sum(self.likelihood(data, extras), axis=1))
            )
            logger.debug(f"Log-likelihood: {last_log_likelihood}  --> {log_likelihood}")

            relative_difference = abs(
                (last_log_likelihood - log_likelihood) / log_likelihood
            )

            has_converged = relative_difference < tolerance
            if callback:
                logger.debug(f"Running callback for iteration {iteration}")
                callback(**locals())

            if has_converged:
                logger.info("Converged")
                return True

            last_log_likelihood = log_likelihood
        else:
            # Iteration finished without encountering a break, meaning it did not converge.
            logger.warning("Did not converge")
            return False

# ============================================================================

    def e_step(self, data, extras=None):
        """Get posterior weights.

         The normalization factor can be used to go back to likelihood x prior, used to calculate log-likelihood.
         """
        logger.debug(f"E-step")

        weights = self.likelihood(data, extras)

        # Avoid impossible classes
        weights[weights < 1e-12] = 1e-12

        normalization_factors = np.sum(weights, axis=1)[:, None]
        weights /= normalization_factors

        number_of_samples, number_of_dimensions = data.shape
        if not np.allclose(data.shape, [number_of_samples, number_of_dimensions]):
            raise ValueError((
                f"Expected data matrix to have shape [number_of_samples x number_of_dimensions], "
                f"but got {data.shape}"
            ))
        if not np.allclose(weights.shape, [number_of_samples, self.number_of_classes]):
            raise ValueError((
                f"Expected weight matrix to have shape [number_of_samples x number_of_dimensions], "
                f"got {weights.shape}"
            ))

        return weights, normalization_factors

    def m_step(self, data, weights, extras=None, iteration=None):
        logger.debug(f"M-step")

        number_of_samples, number_of_dimensions = data.shape

        self.priors = np.sum(weights, axis=0) / number_of_samples
        # if np.allclose(self.priors, 1 / self.number_of_classes):
            # self.priors = np.random.random(self.number_of_classes)

        # Calculate updated parameters
        for m, component in enumerate(self.components):
            component.update_parameters(
                data, weights[:, m], extras, iteration=iteration
            )

    @property
    def priors(self):
        return self._priors

    def pdf(self, x, extras=None):
        return sum(
            component.pdf(x, prior=prior, extras=extras)
            for (prior, component) in zip(self.priors, self.components)
        )

    @priors.setter
    def priors(self, priors):
        """Make sure prior always sum to one."""
        if not priors.shape == (self.number_of_classes,):
            raise ValueError("Priors need to match the number of components")
        priors[priors < 1e-14] = 1e-14
        self._priors = priors / np.sum(priors)
        if np.any(np.isnan(priors)):
            logger.warning("Found NaN in prior")
        if not np.allclose(self._priors.sum(), 1):
            logger.warning("Priors should sum to one")
        logger.debug(f"Updating priors: {np.round(self._priors, 4)}")

    @property
    def number_of_classes(self):
        return len(self.components)

    def save(self, output_path):
        output_path = pathlib.Path(output_path).resolve()
        logger.info(f"Saving model to {output_path}")
        with open(output_path, "w") as f:
            model_dict = self.to_dict()
            toml.dump(model_dict, f)

    def to_dict(self):
        return {
            "priors": self.priors.tolist(),
            "components": [component.to_dict() for component in self.components],
        }

    @staticmethod
    def load(filename):
        with open(filename, "r") as f:
            model_dict = toml.load(f)
        return MixtureModel.from_dict(model_dict)

    @staticmethod
    def from_dict(model_dict):
        priors = model_dict["priors"]

        components = []
        for c_dict in model_dict["components"]:
            component_class = MixtureComponent.REGISTRY[c_dict["type"]]
            c = component_class(**c_dict["parameters"])
            c.label = c_dict.get("label", "")
            components.append(c)
        model = MixtureModel(components=components, priors=priors)
        return model

    def likelihood(self, data, extras=None):
        """
        For numerical stability, it sometimes makes sense to include prior inside the exp().
        So, the implementation leaves it to the pdf to include a prior weight, rather than multiplying the pdf value.
        """

        # data.shape = number_of_samples, number_of_dimensions
        L = np.hstack(
            [
                component.pdf(data, prior=self.priors[m], extras=extras).ravel()[:,None]
                for m, component in enumerate(self.components)
            ]
        )
        return L

    def predict(self, data, extras=None):
        weights = self.predict_soft(data, extras=extras)
        return np.argmax(weights, axis=1)

    def predict_soft(self, data, extras=None):
        weights, normalization_factor = self.e_step(data, extras=extras)
        return weights

    def initialize_from_data(self, data, extras=None):
        for component in self.components:
            component.initialize_from_data(data, extras=extras)


class MixtureComponent(ABC):
    REGISTRY = dict()

    def __init_subclass__(cls, **kwargs):
        # Make sure any sub-classes are added to the registry
        # to support serialization/de-serialization
        if cls not in cls.REGISTRY:
            cls.REGISTRY[cls.__name__] = cls

        if not hasattr(cls, "NAME"):
            cls.NAME = cls.__name__

        if not hasattr(cls, "PARAMETERS"):
            raise NotImplementedError(
                "The component needs to specify the `PARAMETERS` attribute on the class."
            )
        super().__init_subclass__(**kwargs)

    @property
    def label(self):
        return self.__dict__.get("_MixtureComponent__label", "")

    @label.setter
    def label(self, label):
        logger.debug(f"Setting label `{label}` for component {self}")
        self.__label = label

    def to_dict(self):
        return {
            "type": type(self).__name__,
            "dimensions": self.dimensions,
            "parameters": {
                parameter: np.array(self.__getattribute__(parameter)).tolist()
                for parameter in self.PARAMETERS
            },
            "label": self.label,
        }

    def initialize_from_data(self, data, extras=None):
        return

    @abstractmethod
    def pdf(self, data, prior=1.0, extras=None):
        ...

    @abstractmethod
    def update_parameters(self):
        ...

    @property
    @abstractmethod
    def degrees_of_freedom(self):
        ...
