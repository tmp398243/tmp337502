# # Description
# This example shows a simple use case for NormalizingFlowFilters.
#
# First, we import the necessary packages.
using NormalizingFlowFilters
using Random: randn
using Statistics: mean, std
using Test

# Then define the filter.
glow_config = ConditionalGlowOptions()
network = NetworkConditionalGlow(2, glow_config)

optimizer_config = OptimizerOptions(lr=1e-20)
optimizer = create_optimizer(optimizer_config)

device = cpu
training_config = TrainingOptions(num_post_samples=2, noise_lev_y=1e0, noise_lev_x=1e0)

filter = NormalizingFlowFilter(network, optimizer; device, training_config)

# We generate an ensemble.

## N ensemble members from a unit normal.
N = 10
prior_state = randn(Float64, 3, N)
prior_state .-= mean(prior_state; dims=2)
prior_state ./= std(prior_state; dims=2)

## Identity observation operator with no noise.
prior_obs = deepcopy(prior_state) .+ randn(Float64, 3, N)

# Then we assimilate an observation. Here, we just pick an arbitrary one.
y_obs = [0.0, 0.0, 0.0]
posterior = assimilate_data(filter, prior_state, prior_obs, y_obs)

@show mean(posterior; dims=2) std(posterior; dims=2)

# The posterior should have mean 0 and some specific variance, since we don't have any noise in the measurements.

X = reshape(prior_state, (1, 1, size(prior_state, 1), size(prior_state, 2)))
Y = reshape(prior_obs, (1, 1, size(prior_obs, 1), size(prior_obs, 2)))
y_obs_r = reshape(y_obs, (1, 1, size(y_obs, 1), size(y_obs, 2)))

fresh_samples = draw_posterior_samples(
    filter.network_device,
    y_obs_r,
    size(X);
    device=filter.device,
    num_samples=10,
    batch_size=filter.training_config.batch_size,
    log_data=nothing,
)[1, 1, :, :]

@show mean(fresh_samples; dims=2) std(fresh_samples; dims=2)

@test true
