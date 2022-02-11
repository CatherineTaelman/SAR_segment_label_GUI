from scipy.stats import chisquare, norm as standard_normal, chi2, ksone, kstwobign, gaussian_kde
from loguru import logger
from scipy import linalg
import numpy as np

from .base import MixtureModel
from .components import gaussian, gamma, GLIA


def chi_square_goodness_of_fit_score(model, data, extras=None, significance_level=0.01):
    # TODO: Could take in posteriors instead of data to avoid problems with `extras`
    posteriors = model.predict_soft(data, extras=extras)
    number_of_samples, number_of_dimensions = data.shape
    number_of_classes = len(model.components)
    chi_square_test_statistics = np.zeros((number_of_classes, number_of_dimensions), dtype=np.float32)
    is_good_fit = np.zeros((number_of_classes, number_of_dimensions), dtype=bool)
    comboSigLevel = significance_level/(number_of_classes*number_of_dimensions)
    degrees_of_freedom = 2
   
    toDrop = np.zeros((number_of_classes), dtype=np.int)
    
    # TODO: This should be done for each dimension, but for now assume 1D data
    for class_index, (prior, component) in enumerate(zip(model.priors, model.components)):
        inverse_square_root_matrix = linalg.inv(linalg.sqrtm(component.cov))

        # Normalize with mean and covariance of current component.
        # This should make the distribution rougly standard multivariate normal (mean zero and identity covariance),
        # which makes the projection onto each dimension roughly univariate standard normal (assuming good fit).
        normalized_deviations = (data - component.mu) @ inverse_square_root_matrix
        
        # adapt num.bins to keep at least 5 counts per bin.
        Nj = np.nansum(posteriors[:, class_index])
        useBins = np.fmin(60, np.fmax(1, np.floor(Nj/5))).astype(np.integer)
        expected_counts = Nj / useBins
        
        for dimension in range(number_of_dimensions):
            if useBins > (degrees_of_freedom+1):
                edges = standard_normal.ppf(np.linspace(0, 1, useBins))
                observed_counts, _ = np.histogram(
                    normalized_deviations[:, dimension],
                    edges,
                    weights=posteriors[:, class_index]
                )

                # TODO: Revisit this later
                # valid_bins = (observed_counts > 5) & (expected_counts > 5)

                test_statistic = np.nansum((observed_counts - expected_counts) ** 2 / expected_counts)

                threshold = chi2.ppf(1 - comboSigLevel, useBins - 1 - degrees_of_freedom)

            else:
                toDrop[class_index] = 1
                test_statistic = 0.0
                threshold = 1.0
                
            chi_square_test_statistics[class_index, dimension] = test_statistic
            is_good_fit[class_index, dimension] = test_statistic < threshold
            
        logger.debug((
            f"[Class {class_index}]\t "
            f"Chi2 statistic: {chi_square_test_statistics[class_index]}\t "
            f"Threshold: {threshold} {'GOOD FIT' if is_good_fit[class_index, :].all() else 'BAD FIT'}\t "
            f"Bins: {useBins}\t "
            f"Exptd: {expected_counts.round(3)}"
        ))

    return chi_square_test_statistics, is_good_fit, toDrop

#================================================================


def chi_square_goodness_of_fit_score_with_IA(model, data, extras=None, significance_level=0.01):
    # TODO: Could take in posteriors instead of data to avoid problems with `extras`
    posteriors = model.predict_soft(data, extras=extras)
    number_of_samples, number_of_dimensions = data.shape
    number_of_classes = len(model.components)
    chi_square_test_statistics = np.zeros((number_of_classes, number_of_dimensions), dtype=np.float32)
    is_good_fit = np.zeros((number_of_classes, number_of_dimensions), dtype=bool)
    comboSigLevel = significance_level/(number_of_classes*number_of_dimensions)
    degrees_of_freedom = 3
   
    toDrop = np.zeros((number_of_classes), dtype=np.int)

    # TODO: This should be done for each dimension, but for now assume 1D data
    for class_index, (prior, component) in enumerate(zip(model.priors, model.components)):
        inverse_square_root_matrix = linalg.inv(linalg.sqrtm(component.cov))

        # Normalize with mean and covariance of current component.
        # This should make the distribution rougly standard multivariate normal (mean zero and identity covariance),
        # which makes the projection onto each dimension roughly univariate standard normal (assuming good fit).
        
        # NB: simple first version with varmeanfunc, later should do 2-D multinomial.
        normalized_deviations = (data - component.varmeanfunc(extras)) @ inverse_square_root_matrix
        
        # adapt num.bins to keep at least 5 counts per bin.
        Nj = np.nansum(posteriors[:, class_index])
        useBins = np.fmin(50, np.fmax(1, np.floor(Nj/5))).astype(np.integer)
        
        for dimension in range(number_of_dimensions):
            if useBins > (degrees_of_freedom+1):
                edges = standard_normal.ppf(np.linspace(0, 1, useBins))
                observed_counts, _ = np.histogram(
                    normalized_deviations[:, dimension],
                    edges,
                    weights=posteriors[:, class_index]
                )
                bin_centers = (edges[:-1] + edges[1:]) / 2.0

                lower = prior * standard_normal.cdf(edges[:-1])
                upper = prior * standard_normal.cdf(edges[1:])
                expected_counts = number_of_samples * (upper - lower)

                # TODO: Revisit this later
                # valid_bins = (observed_counts > 5) & (expected_counts > 5)

                test_statistic = np.nansum((observed_counts - expected_counts) ** 2 / expected_counts)

                threshold = chi2.ppf(1 - comboSigLevel, useBins - 1 - degrees_of_freedom)
            
            else:
                toDrop[class_index] = 1
                test_statistic = 0.0
                threshold = 1.0
            
            
            chi_square_test_statistics[class_index, dimension] = test_statistic

            is_good_fit[class_index, dimension] = test_statistic < threshold

        logger.debug((
            f"[Class {class_index}]\t "
            f"Chi2 statistic: {chi_square_test_statistics[class_index]}\t "
            f"Threshold: {threshold} {'GOOD FIT' if is_good_fit[class_index, :].all() else 'BAD FIT'}\t "
            f"Bins: {useBins}"
        ))

    return chi_square_test_statistics, is_good_fit, toDrop

#================================================================


def chi_square_goodness_of_fit_score_with_IA_2D(model, data, extras=None, significance_level=0.01):
    # TODO: Could take in posteriors instead of data to avoid problems with `extras`
    posteriors = model.predict_soft(data, extras=extras)
    number_of_samples, number_of_dimensions = data.shape
    number_of_classes = len(model.components)
    chi_square_test_statistics = np.zeros((number_of_classes, number_of_dimensions), dtype=np.float32)
    is_good_fit = np.zeros((number_of_classes, number_of_dimensions), dtype=bool)
    comboSigLevel = significance_level/(number_of_classes*number_of_dimensions)
    degrees_of_freedom = 3
   
    toDrop = np.zeros((number_of_classes), dtype=np.int)
    IAmin = 15
    IAmax = 50
    IAeval = np.linspace(IAmin, IAmax, num=70)   #  70 should give every 0.5 degrees.
        

    # TODO: This should be done for each dimension, but for now assume 1D data
    for class_index, (prior, component) in enumerate(zip(model.priors, model.components)):
        inverse_square_root_matrix = linalg.inv(linalg.sqrtm(component.cov))

        # Normalize with mean and covariance of current component.
        # This should make the distribution rougly standard multivariate normal (mean zero and identity covariance),
        # which makes the projection onto each dimension roughly univariate standard normal (assuming good fit).
        
        # NB: simple first version with varmeanfunc, later should do 2-D multinomial.
        normalized_deviations = (data - component.varmeanfunc(extras)) @ inverse_square_root_matrix
        
        # adapt num.bins to keep at least 5 counts per bin.
        Nj = number_of_samples*model.priors[class_index]  #  np.nansum(posteriors[:, class_index])
        totB = np.fmin(60, np.fmax(1, np.floor(Nj/15))).astype(np.integer)   #  aim for at least 15 counts per bin.
        binsA = int(np.fmax(1, np.floor(np.sqrt(totB/3))))
        binsI = int(np.fmax(1, np.floor(totB/binsA)))    #  want more bins in Intensity than IA.
        bins = binsI*binsA
        expected_counts = Nj / bins
        threshold = chi2.ppf(1 - comboSigLevel, bins - 1 - degrees_of_freedom)
        logger.debug("#{:3d}, Nj = {:8.3f}, binsI = {:2d}, binsA = {:2d}".format(class_index, np.around(Nj, decimals=3), binsI, binsA))

        if bins > 1: # (degrees_of_freedom+1):
            edgesI = standard_normal.ppf(np.arange(1.,binsI+1.)/binsI)
            evalPs = np.linspace(0, 1, binsA, endpoint=False)[1:]
            # now find empirical cdf of SubIA
            parzenj = gaussian_kde(extras["IA"], bw_method=None, weights=posteriors[:, class_index])
            ecdf = np.cumsum(parzenj(IAeval), dtype=np.float32)
            ecdf = ecdf/np.nanmax(ecdf)   #  normalise cumsum, just in case.
            if ecdf[0] > 0:
                ecdf[1] += ecdf[0]	# push it to next not to lose the count.
                ecdf[0] = 0.0
            fZeros = np.nonzero(ecdf == 0.0)[0]
            if len(fZeros) > 1:
                ecdf[fZeros[0:-1]] = np.nan # remove multiple zeroes.
            fOnes = np.nonzero(ecdf == 1.0)[0]
            if len(fOnes) > 1:
                ecdf[fOnes[1:]] = np.nan # remove multiple ones.
            edgesA = np.concatenate(([IAmin], np.interp(evalPs, ecdf[np.isfinite(ecdf)], IAeval[np.isfinite(ecdf)]), [IAmax]), axis=0)
            

            for dimension in range(number_of_dimensions):
                
                #logger.debug("IA "+str(extras["IA"].shape)+"  ndev.d "+str(normalized_deviations[:, dimension].shape)+"  posts "+str(posteriors[:, class_index].shape))
                
                edges = standard_normal.ppf(np.linspace(0, 1, bins))
                observed_counts, _, _ = np.histogram2d(extras["IA"],
                    normalized_deviations[:, dimension],
                    bins=(edgesA, edgesI),
                    weights=posteriors[:, class_index]
                )

                test_statistic = np.nansum((observed_counts - expected_counts) ** 2 / expected_counts)

                chi_square_test_statistics[class_index, dimension] = test_statistic

                is_good_fit[class_index, dimension] = test_statistic < threshold


        else:
            toDrop[class_index] = 1
            threshold = 1.0
            chi_square_test_statistics[class_index, :] = np.tile((0.0), (1, number_of_dimensions))
            is_good_fit[class_index, :] = True
            
            
        logger.debug((
            f"[Class {class_index}]\t "
            f"Chi2 statistic: {chi_square_test_statistics[class_index]}\t "
            f"Threshold: {threshold} {'GOOD FIT' if is_good_fit[class_index, :].all() else 'BAD FIT'}\t "
            f"Bins: {bins}\t "
            f"Exptd: {expected_counts.round(3)} "
        ))

    return chi_square_test_statistics, is_good_fit, toDrop

#================================================================

def fit_GLIA_noise_per_subswath(
        data,
        max_classes=3,
        extras=None,
        fit_options=dict(
            adaptive_noise=False,
            max_iterations=100,
            tolerance=1e-5,
            callback=None,
        )
):
    SSLs = np.zeros((extras["Nsw"]))
    for swi in range(extras["Nsw"]):
        logger.info("Fitting sub-swath " + str(swi+1))
        swiDat = data[(extras["SSM"]==swi+1), :]
        swiExtras = {"IA":extras["IA"][(extras["SSM"]==swi+1)], "NF":extras["NF"][(extras["SSM"]==swi+1), :], "azNF":extras["azNF"][(extras["SSM"]==swi+1), :], "SSM":extras["SSM"][(extras["SSM"]==swi+1)], "NLS":np.array([1.0]), "Nsw":1}

        MoG = fit_mixture_of_GLIA(swiDat, max_classes=max_classes, extras=swiExtras, fit_options=dict(adaptive_noise=fit_options["adaptive_noise"], max_iterations=100, tolerance=1E-8, callback=None,))
        SSLs[swi] = extras["NLS"][0]
        
    #logger.info("SSLs = " + str(SSLs))
    return SSLs

#================================================================

def fit_mixture_of_GLIA(
        data,
        max_classes=3,
        extras=None,
        fit_options=dict(
            max_iterations=100,
            tolerance=1e-5,
            callback=None,
        )
):
    dimensions = data.shape[1]
    

    # Start with a single component
    model = MixtureModel([GLIA.GLIA(dimensions=dimensions)])
    model.initialize_from_data(data, extras=extras)

    # Keep splitting until we get a good fit
    for iteration in range(max_classes-1):
        logger.debug("GoF loop iteration " + str(iteration) + "  / " + str(max_classes))
        # Fit current model
        model.fit(data, **fit_options, extras=extras, reinitialize=False)

        # Get goodness of fit for each class
        scores, good_fit, to_drop = chi_square_goodness_of_fit_score_with_IA_2D(model, data, extras=extras)
            
        if np.all(good_fit):
            logger.success("All good fits")
            if (to_drop.any()):
                logger.warning("Dropping {0} empty classes.".format(np.sum(to_drop)))
                model.components = model.components[to_drop == 0]
                model.priors = model.priors[to_drop == 0]
            break
        
        if (iteration < (max_classes -2)):  #   don't split the last round. just fit and test it.
            # Split worst component into two new components, where each has half the std.dev.
            # and the mean is shifted up/down by half the standard deviation
            worst_class, worst_dimension = np.unravel_index(scores.argmax(), scores.shape)
        
            logger.debug("Splitting worst class {0}".format(worst_class))

            component_to_split = model.components[worst_class]
            mu = component_to_split.mu
            beta = component_to_split.beta
            cov = component_to_split.cov

            square_root_matrix = linalg.sqrtm(component_to_split.cov)
            shift = np.zeros(dimensions)
            shift[worst_dimension] = 0.5
            shift = shift @ square_root_matrix

            new_component_lower = GLIA.GLIA(dimensions=dimensions, mu=mu - shift, beta=beta, cov=0.5 * cov)
            new_component_upper = GLIA.GLIA(dimensions=dimensions, mu=mu + shift, beta=beta, cov=0.5 * cov)
        
            if (to_drop.any()):
                logger.warning("to_drop {0} empty classes. {1}".format(np.sum(to_drop), str(to_drop)))

            # Create updated model, but preserve priors for the classes that are not being split
            currNum = model.number_of_classes
            model = MixtureModel(
                    list(model.components[mc] for mc in range(worst_class) if not to_drop[mc]) + 
                    [new_component_lower, new_component_upper] + 
                    list(model.components[mc] for mc in range(worst_class+1, currNum, 1) if not to_drop[mc])
                ,
                priors=
                     list(model.priors[mc] for mc in range(worst_class) if not to_drop[mc]) + 
                    [0.5 * model.priors[worst_class], 0.5 * model.priors[worst_class]] + 
                    list(model.priors[mc] for mc in range(worst_class+1, currNum, 1) if not to_drop[mc])
            
            )
        else:
            if (to_drop.any()):
                logger.warning("Dropping {0} empty classes.".format(np.sum(to_drop)))
                model.components = list(model.components[mc] for mc in range(model.number_of_classes) if not to_drop[mc])
                model.priors = model.priors[to_drop == 0]

    else:
        logger.warning("Did not converge")

    return model
    
#================================================================


def fit_mixture_of_gaussians(
        data,
        max_classes=3,
        extras=None,
        fit_options=dict(
            max_iterations=100,
            tolerance=1e-5,
            callback=None,
        )
):
    dimensions = data.shape[1]

    # Start with a single component
    model = MixtureModel([gaussian.MultivariateGaussian(dimensions=dimensions)])
    model.initialize_from_data(data, extras=extras)

    # Keep splitting until we get a good fit
    for iteration in range(max_classes-1):
        # Fit current model
        model.fit(data, **fit_options, extras=extras, reinitialize=False)

        # Get goodness of fit for each class
        scores, good_fit, to_drop = chi_square_goodness_of_fit_score(model, data, extras=extras)
        
        if np.all(good_fit):
            logger.success("All good fits")
            break

        logger.debug("Splitting worst class")

        # Split worst component into two new components, where each has half the std.dev.
        # and the mean is shifted up/down by half the standard deviation
        worst_class, worst_dimension = np.unravel_index(scores.argmax(), scores.shape)
        component_to_split = model.components[worst_class]
        mu = component_to_split.mu
        cov = component_to_split.cov

        square_root_matrix = linalg.sqrtm(component_to_split.cov)
        shift = np.zeros(dimensions)
        shift[worst_dimension] = 0.5
        shift = shift @ square_root_matrix

        new_component_lower = gaussian.MultivariateGaussian(dimensions=dimensions, mu=mu - shift, cov=0.5 * cov)
        new_component_upper = gaussian.MultivariateGaussian(dimensions=dimensions, mu=mu + shift, cov=0.5 * cov)
 
        if (to_drop.any()):
            logger.warning("to_drop {0} empty classes. {1}".format(np.sum(to_drop), str(to_drop)))

        # Create updated model, but preserve priors for the classes that are not being split
        currNum = model.number_of_classes
        model = MixtureModel(
                (list(model.components[mc] for mc in range(worst_class) if not to_drop[mc]) + 
                [new_component_lower, new_component_upper] + 
                list(model.components[mc] for mc in range(worst_class+1, currNum, 1) if not to_drop[mc]))
            ,
            priors=
                 (list(model.priors[mc] for mc in range(worst_class) if not to_drop[mc]) + 
                [0.5 * model.priors[worst_class], 0.5 * model.priors[worst_class]] + 
                list(model.priors[mc] for mc in range(worst_class+1, currNum, 1) if not to_drop[mc]))
            
        )

    else:
        logger.warning("Did not converge")

    return model

#================================================================


def automatic_mixture_of_univariate_gammas(
        data, alpha=4.4, max_classes=3, fit_options=dict(max_iterations=20)
):
    model = MixtureModel([RightTruncatedUnivariateGamma(alpha=alpha, fix_alpha=True)])
    for component in model.components:
        component.initialize_from_data(data)

    callback = test_gamma.plot_univariate()

    for iteration in range(max_classes):
        model.fit(data, **fit_options, callback=callback)

        scores, good_fit, to_drop = chi_square_goodness_of_fit_score(model, data, extras=extras)
        if np.all(good_fit):
            logger.success("All good fits")
            break

        worst_component, *components_to_keep = list(
            [
                c
                for (_, c) in reversed(
                sorted(zip(scores, model.components), key=lambda pair: pair[0])
            )
            ]
        )

        logger.warning("Splitting worst class")
        model = MixtureModel(
            [
                *components_to_keep,
                RightTruncatedUnivariateGamma(
                    mu=0.9 * worst_component.mu, alpha=alpha, fix_alpha=True
                ),
                RightTruncatedUnivariateGamma(
                    mu=1.1 * worst_component.mu, alpha=alpha, fix_alpha=True
                ),
            ]
        )
    else:
        logger.warning("Did not converge")

    return model
