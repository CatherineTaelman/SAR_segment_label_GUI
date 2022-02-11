def test_estimate_truncated():
    means, standard_deviations = [-5, 3], [3, 3]
    data = np.hstack(
        [
            simulate_gaussian(400, mu, sigma)
            for mu, sigma in zip(means, standard_deviations)
        ]
    ).T

    lower, upper = -10, 10
    truncated_data = data[(lower <= data) & (data <= upper)]

    model = MixtureModel(
        components=[TruncatedGamma(lower, upper), TruncatedGamma(lower, upper)]
    )
    model.fit(truncated_data[:, None], max_iterations=500, callback=make_callback())

    # assert np.allclose(estimated_mu, mu)
    # assert np.allclose(estimated_sigma, sigma)
    # plt.show(block=True)
    plt.show(block=False)
    plt.pause(2)


@pytest.mark.skip(reason="Just an idea")
def test_mixture_of_truncated_gammas():
    # Fit a mixture of 3 truncated Gamma distributions with specified shape parameter
    enl = 12
    model = MixtureModel(
        components=[Truncate(Gamma(shape=enl), left=2, right=100) for _ in range(3)]
    )
