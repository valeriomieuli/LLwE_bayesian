# LLwE_bayesian

Do not consider autoencders.py and experts.py.

bayesian_models.py trains and saves a bayesian model according to the name of the dataset passed as input parameter.

evaluate_uncertainty.py loads a bayesian model according to the the name of the dataset passed as input and then compute uncertainties (std, aleatoric and epistemic) for all datasets. These values are then plotted on some scatterplots.
