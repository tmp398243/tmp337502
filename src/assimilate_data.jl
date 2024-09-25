using Flux: cpu, gpu
using LinearAlgebra: norm
using Random: randn

export assimilate_data, draw_posterior_samples

function draw_posterior_samples(
    G, y, X, Y, size_x; device=gpu, num_samples, batch_size, log_data=nothing
)
    X_forward = device(randn(Float64, size_x[1:(end - 1)]..., batch_size))
    y_r = reshape(cpu(y), 1, 1, :, 1)
    Y_train_latent_repeat = device(repeat(y_r, 1, 1, 1, batch_size))
    Zx_fixed_train, Zy_fixed_train, _ = G.forward(X_forward, Y_train_latent_repeat)

    X_post = zeros(Float32, size_x[1:(end - 1)]..., num_samples)
    for i in 1:div(num_samples, batch_size)
        #ZX_noise_i = randn(Float32, size_x[1:end-1]...,batch_size)|> device
        X_forward_i = X[:, :, :, ((i - 1) * batch_size + 1):(i * batch_size)]
        Y_forward_i = Y[:, :, :, ((i - 1) * batch_size + 1):(i * batch_size)]
        Zx_fixed_train_i, _, _ = G.forward(device(X_forward_i), device(Y_forward_i))
        X_post[:, :, :, ((i - 1) * batch_size + 1):(i * batch_size)] =
            cpu(G.inverse(Zx_fixed_train_i, Zy_fixed_train))
    end
    return X_post
end


function draw_posterior_samples(
    G, y, size_x; device=gpu, num_samples, batch_size, log_data=nothing
)
    X_forward = device(randn(Float64, size_x[1:(end - 1)]..., batch_size))
    y_r = reshape(cpu(y), 1, 1, :, 1)
    Y_train_latent_repeat = device(repeat(y_r, 1, 1, 1, batch_size))
    Zx_fixed_train, Zy_fixed_train, _ = G.forward(X_forward, Y_train_latent_repeat)
    @show Zy_fixed_train y

    X_post = zeros(Float32, size_x[1:(end - 1)]..., num_samples)
    for i in 1:div(num_samples, batch_size)
        ZX_noise_i = randn(Float64, size_x[1:end-1]...,batch_size)|> device
        X_post[:, :, :, ((i - 1) * batch_size + 1):(i * batch_size)] =
            cpu(G.inverse(ZX_noise_i, Zy_fixed_train))
    end
    return X_post
end

"""

- `prior_state` has shape `(s..., N)` for state shape `s` and number of ensemble members `N`.
- `prior_obs` has shape `(r..., N)` for observation shape `r` and number of ensemble members `N`.
- `y_obs` has shape `r` for observation shape `r`.

"""
function assimilate_data(
    filter::NormalizingFlowFilter,
    prior_state::AbstractArray,
    prior_obs::AbstractArray,
    y_obs,
    log_data=nothing,
)
    X = prior_state
    Y = prior_obs

    X = reshape(X, (1, 1, size(X, 1), size(X, 2)))
    Y = reshape(Y, (1, 1, size(Y, 1), size(Y, 2)))

    train_network!(filter, X, Y; log_data)

    y_obs = reshape(y_obs, (1, 1, size(y_obs, 1), size(y_obs, 2)))
    X = draw_posterior_samples(
        filter.network_device,
        y_obs,
        X,
        Y,
        size(X);
        device=filter.device,
        num_samples=size(X, 4),
        batch_size=filter.training_config.batch_size,
        log_data,
    )
    posterior = X[1, 1, :, :]
    return posterior
end
