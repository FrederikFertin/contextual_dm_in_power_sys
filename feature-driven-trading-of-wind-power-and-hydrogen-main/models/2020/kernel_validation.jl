using Gurobi
using JuMP
using DataFrames
using CSV
using Distances
using LinearAlgebra
using NearestNeighbors
using Plots

function get_opt_z(X, Y, W, N, F)
    # period length is the amount of timesteps used for training
    # bidding_start is the timestep for the first bid, it is expected that 24 bids are needed

    if (N % 24 != 0)
        throw(ErrorException("Training period must be a multiple of 24 hours!"))
    end

    periods = collect(1:N)
    days = []
    n_days = Int(N / 24)
    for i in collect(1:n_days)
        day_offset = (i - 1) * 24
        push!(days, collect(1+day_offset:24+day_offset))
    end

    #Declare Gurobi model
    initial_plan = Model(Gurobi.Optimizer)

    #Definition of variables
    @variable(initial_plan, 0 <= E_DW[t in periods])
    @variable(initial_plan, 0 <= E_UP[t in periods])
    @variable(initial_plan, b[t in periods], Bin) # Binary variable indicating if we are deficit_settle (1) or surplus_settle (0)
    @variable(initial_plan, E_settle[t in periods])
    @variable(initial_plan, qF[1:(F+1), 1:24])
    @variable(initial_plan, qH[1:(F+1), 1:24])

    @variable(initial_plan, 0 <= hydrogen[t in periods])
    @variable(initial_plan, forward_bid[t in periods])

    #Maximize profit
    @objective(initial_plan, Max,
        sum( W[t] * 
            (lambda_F[t] * forward_bid[t]
            + lambda_H * hydrogen[t]
            + lambda_DW[t] * E_DW[t]
            - lambda_UP[t] * E_UP[t])
            for t in periods
        ) 
    )


    #Max capacity
    @constraint(initial_plan, wind_capacity_up[t in periods], forward_bid[t] <= max_wind_capacity)
    @constraint(initial_plan, wind_capacity_dw[t in periods], forward_bid[t] >= -max_elec_capacity)
    @constraint(initial_plan, elec_capacity[t in periods], hydrogen[t] <= max_elec_capacity)

    # Power surplus == POSITIVE, deficit == NEGATIVE
    @constraint(initial_plan, settlement[t in periods], Y[t] - forward_bid[t] - hydrogen[t] == E_settle[t])

    @constraint(initial_plan, surplus_settle1[t in periods], E_DW[t] >= E_settle[t])
    @constraint(initial_plan, surplus_settle2[t in periods], E_DW[t] <= E_settle[t] + M * b[t])
    @constraint(initial_plan, surplus_settle3[t in periods], E_DW[t] <= M * (1 - b[t]))

    @constraint(initial_plan, deficit_settle1[t in periods], E_UP[t] >= -E_settle[t])
    @constraint(initial_plan, deficit_settle2[t in periods], E_UP[t] <= -E_settle[t] + M * (1 - b[t]))
    @constraint(initial_plan, deficit_settle3[t in periods], E_UP[t] <= M * (b[t]))

    for day in days
        @constraint(initial_plan, sum(hydrogen[t] for t in day) >= min_production)
        for t in day
            index = mod(t, 24)
            if (index == 0)
                index = 24
            end
            @constraint(initial_plan, forward_bid[t] == sum(qF[i, index] * X[t, i] for i in 1:F) + qF[F+1, index])
            @constraint(initial_plan, hydrogen[t] == sum(qH[i, index] * X[t, i] for i in 1:F) + qH[F+1, index])
        end
    end

    optimize!(initial_plan)

    return [[value.(qF[i, :]) for i in 1:F+1], [value.(qH[i, :]) for i in 1:F+1]]
end

function gaussian_kernel(x_t,x_u,sigma=0.51)
    # Finds the weight of a data point x_t given a fitting point x_u
    x_m = x_t .- transpose(x_u)
    dists = map(norm, eachslice(x_m, dims=1))
    return  exp.(-dists ./ (2*sigma^2) )
end

function cf_fit(X, y)
    """ Closed form weighted fit """
    XT = transpose(X)
    return inv(XT * X) * XT * y
end

function cf_weighted_fit(X, y, W)
    """ Closed form weighted fit """
    XT = transpose(X)
    return inv(XT * W * X) * XT * W * y
end

function cf_predict(beta, X)
    """ Closed form fit """
    y_pred = transpose(X) * beta
    return y_pred
end

function gaussian_kernel_fit_predict(X_train, y_train, X_test, sigma = 0.51, message=true)    
    # Number of test points 
    M = size(X_test)[1]
    
    # Initialize predictions 
    y_pred = Matrix{Float64}(undef,M,2)
    
    # Loop through test points 
    for i in 1:M
        if message
            println("Test point $i / $M")
        end
        
        W_vector = gaussian_kernel(X_train, X_test[i,:], sigma)
        
        W = diagm(W_vector)
        
        beta = cf_weighted_fit(X_train, y_train, W)
        
        x = cf_predict(beta,X_test[i,:])
        y_pred[i,1] = min(10,max(x[1],-10))
        y_pred[i,2] = min(10,max(x[2],0))
    end
    return y_pred 
end

function knn_fit_predict(X_train, y_train, X_test, k::Int)
    k = max(k,3)
    # Number of test points 
    M = size(X_test)[1]
    # Number of train points
    N, F = size(X_train)
    # Initialize predictions
    d_z = 2
    z = Matrix{Float64}(undef, M, d_z)

    dists = pairwise(Euclidean(), transpose(X_train), transpose(X_test), dims=2)

    for i = 1:convert(Int,M/24)
        w = zeros(N)
        day_slice = (i-1)*24+1:i*24
        for ii = day_slice
            dist = dists[:, ii]
            near = sortperm(vec(dist))
            for j = 1:k
                w[near[j]] += 1.0
            end
        end
        w = w/24        

        q = get_opt_z(X_train, y_train, w, N, F)
        
        for ii = day_slice
            for z_act = 1:d_z
                z[ii, z_act] = sum(q[z_act][jj][ii%24+1]*X_test[ii,jj] for jj=1:F) + q[z_act][F+1][ii%24+1]
            end
        end
        # x = cf_predict(cf_fit(X_train[idxs[i],:], y_train[idxs[i],:]), X_test[i,:])
        # z[i,1] = min(10,max(x[1],-10))
        # z[i,2] = min(10,max(x[2],0))
    end
    return z
end

function gaussian_fit_predict(X_train, y_train, X_test)
    
    # Number of test points 
    M = size(X_test)[1]
    # Number of train points
    N, F = size(X_train)
    # Initialize predictions
    d_z = 2
    z = Matrix{Float64}(undef, M, d_z)

    dists = pairwise(SqEuclidean(), transpose(X_train), transpose(X_test), dims=2)

    for i = 1:convert(Int,M/24)
        w = zeros(N)
        day_slice = (i-1)*24+1:i*24
        for ii = day_slice
            dist = dists[:, ii]
            w += exp.(-dist/3)
        end
        w = w/24        

        q = get_opt_z(X_train, y_train, w, N, F)
        
        for ii = day_slice
            for z_act = 1:d_z
                z[ii, z_act] = sum(q[z_act][jj][ii%24+1]*X_test[ii,jj] for jj=1:F) + q[z_act][F+1][ii%24+1]
            end
        end
        # x = cf_predict(cf_fit(X_train[idxs[i],:], y_train[idxs[i],:]), X_test[i,:])
        # z[i,1] = min(10,max(x[1],-10))
        # z[i,2] = min(10,max(x[2],0))
    end
    return z
end

include(joinpath(pwd(), "data_loader_2020.jl"))
#include(joinpath(pwd(), "models/2020/GA.jl"))
# x = all_data[:, ["Offshore DK2", "Offshore DK1", "Onshore DK2", "Onshore DK1", "production_FC", "forward_RE"]]
# x_rf = all_data[:, ["production_FC", "forward_RE"]]
# x_fm = [production_FC_fm, forward_RE]
include(joinpath(pwd(), "data_export.jl"))
z_opt_train = Matrix(DataFrame(CSV.File(joinpath(pwd(), "results/2020/optimal_everything_training_data.csv"))))
z_opt_test = Matrix(DataFrame(CSV.File(joinpath(pwd(), "results/2020/optimal_everything.csv"))))

n = 365
test_points = 24*n

X = x_rf
Y = E_real

x_train = Matrix(X[1:8760,:])
x_test = Matrix(X[8761:8760+test_points,:])
y_train = Matrix(Y[1:8760,:])
y_test = Matrix(Y[8761:8760+test_points,:])

k=100
y_pred_knn = knn_fit_predict(x_train, y_train, x_test, k)
plot(y_pred_knn, labels=["$(k)-kNN forward" "$(k)-kNN hydrogen"])

y_pred_gaussian = gaussian_fit_predict(x_train, y_train, x_test)
plot!(y_pred_gaussian, labels=["Gaussian forward" "Gaussian hydrogen"])
plot!(z_opt_test[1:test_points,:], labels=["Opt. forward" "Opt. hydrogen"])
savefig(joinpath(pwd(), "results/2020/kernel_based_model_predictions.png"))

data = [
    y_pred_knn[:,1],
    y_pred_knn[:,2]
]
names = [
    "forward bid",
    "hydrogen production"
]
filename = "kernels/bids_$k KNN year HAPD"
typed_dataseries = [[data[1][t] for t = 1:length(data[1])], [data[2][t] for t = 1:length(data[1])]]
df = createDF(typed_dataseries, names)
export_dataframe(df, filename)
print("\n\nExported file: $filename\n\n")

data = [
    y_pred_gaussian[:,1],
    y_pred_gaussian[:,2]
]
names = [
    "forward bid",
    "hydrogen production"
]
filename = "kernels/bids_gaussian year HAPD"
typed_dataseries = [[data[1][t] for t = 1:length(data[1])], [data[2][t] for t = 1:length(data[1])]]
df = createDF(typed_dataseries, names)
export_dataframe(df, filename)
print("\n\nExported file: $filename\n\n")
