# # Description
# This example shows a simple use case for NormalizingFlowFilters.
#
# First, we import the necessary packages.

using Pkg: Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate()

using CairoMakie
using NormalizingFlowFilters
using Random: randn
using Statistics: mean, std
using Test
using Pkg: Pkg

# Then define the filter.
glow_config = ConditionalGlowOptions()
network = NetworkConditionalGlow(2, glow_config)

optimizer_config = OptimizerOptions(; lr=1e-3)
optimizer = create_optimizer(optimizer_config)

device = cpu
training_config = TrainingOptions(;
    n_epochs=32, num_post_samples=4, noise_lev_y=1e-1, noise_lev_x=1e-1, batch_size=16
)

filter = NormalizingFlowFilter(network, optimizer; device, training_config)

# We generate an ensemble.

## N ensemble members from a unit normal.
N = 100
prior_state = randn(Float64, 3, N)
prior_state .-= mean(prior_state; dims=2)
prior_state ./= std(prior_state; dims=2)

## Identity observation operator with no noise.
prior_obs = deepcopy(prior_state) .+ randn(Float64, 3, N)

# Then we assimilate an observation. Here, we just pick an arbitrary one.
y_obs = [0.0, 0.0, 0.0]
log_data = Dict{Symbol,Any}()
posterior = assimilate_data(filter, prior_state, prior_obs, y_obs, log_data)

@show mean(posterior; dims=2) std(posterior; dims=2)

# The posterior should have mean 0 and some TBD variance.

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
)[
    1, 1, :, :,
]

@show mean(fresh_samples; dims=2) std(fresh_samples; dims=2)

@test true

# Plot some training metrics.

common_kwargs = (; linewidth=3)

fig = Figure()

ax = Axis(fig[1, 1])

misfit_train = log_data[:network_training][:training][:loss]
misfit_test = log_data[:network_training][:testing][:loss]

## The training metrics are recorded for each batch, but test metrics are computed each
## epoch.
num_batches = length(misfit_train)
num_epochs = length(misfit_test)
batches_per_epochs = div(num_batches, num_epochs)

test_epochs = 1:num_epochs
train_epochs = (1:num_batches) ./ batches_per_epochs

lines_train = lines!(ax, train_epochs, misfit_train; label="train", common_kwargs...)
lines_test = lines!(ax, test_epochs, misfit_test; label="test", common_kwargs...)

ax.xlabel = "epoch number"
ax.ylabel = "loss: 2-norm"
fig[1, end + 1] = Legend(fig, ax; labelsize=14, unique=true)

ax.title = "Batch size: $(training_config.batch_size)"

ax = Axis(fig[2, 1])

x = log_data[:network_training][:training][:logdet]
lines!(ax, train_epochs, x; color=lines_train.color, common_kwargs...)

x = log_data[:network_training][:testing][:logdet]
lines!(ax, test_epochs, x; color=lines_test.color, common_kwargs...)

ax.xlabel = "epoch number"
ax.ylabel = "loss: log determinant"

fig
