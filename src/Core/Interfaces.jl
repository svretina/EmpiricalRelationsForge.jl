abstract type AbstractRegressionMethod end
abstract type AbstractFittedModel end
abstract type AbstractValidationStrategy end

"""Fit a regression method to a scientific dataset."""
function fit end

"""Evaluate a fitted model on observations."""
function predict end

"""Return a human-readable representation of a fitted relation."""
function formula end

"""Return fitted coefficients in deterministic basis order."""
function coefficients end

"""Validate a regression method or fitted model."""
function validate end
