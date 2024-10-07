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

using PairPlots: PairPlots, pairplot

# Then define the filter.
glow_config = ConditionalGlowOptions()
network = NetworkConditionalGlow(2, glow_config)

optimizer_config = OptimizerOptions(; lr=1e-3)
optimizer = create_optimizer(optimizer_config)

device = cpu
training_config = TrainingOptions(;
    n_epochs=32, num_post_samples=2^4, noise_lev_y=1e-3, noise_lev_x=1e-3, batch_size=2 ^ 5, validation_perc=(1 - 2 ^ -2)
)

filter = NormalizingFlowFilter(network, optimizer; device, training_config)

# We generate an ensemble.

## N ensemble members from a unit normal.
N = 2 ^ 8
prior_state = randn(Float64, 3, N)
prior_state .-= mean(prior_state; dims=2)
prior_state ./= std(prior_state; dims=2)

to_table(a) = (;
    x = a[1, :],
    y = a[2, :],
    z = a[3, :],
)
table_prior_state = to_table(prior_state)

fig = pairplot(
    table_prior_state => (
        PairPlots.Hist(colormap=:Blues),
        PairPlots.MarginDensity(bandwidth=0.2, color=RGBf((49,130,189)./255 ...)),
        PairPlots.TrendLine(color=:red),
        PairPlots.Correlation(),
    ),
    PairPlots.Truth(
        to_table(mean(prior_state; dims=2)[:, 1]),
        label="Mean Values",
        color=(:black, 0.5),
        linewidth=4
    ),
)
supertitle = Label(fig[0, :], "prior state", fontsize=30)
if isinteractive()
    display(fig)
else
    fig
end

# Apply observation operator.

## Identity observation operator with some noise.
prior_obs = 0.5 .* deepcopy(prior_state) .+ 0.5 .* randn(Float64, 3, N)

table_prior_obs = to_table(prior_obs)

fig = pairplot(
    table_prior_obs => (
        PairPlots.Hist(colormap=:Blues),
        PairPlots.MarginDensity(bandwidth=0.2, color=RGBf((49,130,189)./255 ...)),
        PairPlots.TrendLine(color=:red),
        PairPlots.Correlation(),
    ),
    PairPlots.Truth(
        to_table(mean(prior_obs; dims=2)[:, 1]),
        label="Mean Values",
        color=(:black, 0.5),
        linewidth=4
    ),
)
supertitle = Label(fig[0, :], "prior observation", fontsize=30)
if isinteractive()
    display(fig)
else
    fig
end

# Then we assimilate an observation. Here, we just pick an arbitrary one.
y_obs = [0.0, 0.0, 0.0]
log_data = Dict{Symbol,Any}()
posterior = assimilate_data(filter, prior_state, prior_obs, y_obs, log_data)

# Visualize conditionally normalized state.
X = prior_state
Y = prior_obs
X = reshape(X, (1, 1, size(X, 1), size(X, 2)))
Y = reshape(Y, (1, 1, size(Y, 1), size(Y, 2)))
Z = normalize_samples(
    filter.network_device,
    X,
    Y,
    size(X);
    device=filter.device,
    num_samples=N,
    batch_size=filter.training_config.batch_size,
)
Z = Z[1, 1, :, :]

table_Z = to_table(Z)

fig = pairplot(
    table_Z => (
        PairPlots.Hist(colormap=:Blues),
        PairPlots.MarginDensity(bandwidth=0.2, color=RGBf((49,130,189)./255 ...)),
        PairPlots.TrendLine(color=:red),
        PairPlots.Correlation(),
    ),
    PairPlots.Truth(
        to_table(mean(Z; dims=2)[:, 1]),
        label="Mean Values",
        color=(:black, 0.5),
        linewidth=4
    ),
)
supertitle = Label(fig[0, :], "latent state", fontsize=30)
if isinteractive()
    display(fig)
else
    fig
end

# Visualize posterior.
@show mean(posterior; dims=2) std(posterior; dims=2)

table_posterior = to_table(posterior)

fig = pairplot(
    table_posterior => (
        PairPlots.Hist(colormap=:Blues),
        PairPlots.MarginDensity(bandwidth=0.2, color=RGBf((49,130,189)./255 ...)),
        PairPlots.TrendLine(color=:red),
        PairPlots.Correlation(),
    ),
    PairPlots.Truth(
        to_table(mean(posterior; dims=2)[:, 1]),
        label="Mean Values",
        color=(:black, 0.5),
        linewidth=4
    ),
)
supertitle = Label(fig[0, :], "posterior state", fontsize=30)
if isinteractive()
    display(fig)
else
    fig
end

# The posterior should have mean 0 and some TBD variance.

X = reshape(prior_state, (1, 1, size(prior_state, 1), size(prior_state, 2)))
Y = reshape(prior_obs, (1, 1, size(prior_obs, 1), size(prior_obs, 2)))
y_obs_r = reshape(y_obs, (1, 1, size(y_obs, 1), size(y_obs, 2)))

fresh_samples = draw_posterior_samples(
    filter.network_device,
    y_obs_r,
    size(X);
    device=filter.device,
    num_samples=32,
    batch_size=filter.training_config.batch_size,
    log_data=nothing,
)[
    1, 1, :, :,
]

@test true;

@show mean(fresh_samples; dims=2) std(fresh_samples; dims=2)

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

supertitle = Label(fig[0, :], "Training log", fontsize=30)
if isinteractive()
    display(fig)
else
    fig
end
