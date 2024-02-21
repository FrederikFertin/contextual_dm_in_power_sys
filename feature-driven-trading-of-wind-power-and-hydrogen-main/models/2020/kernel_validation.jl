using Gurobi
using JuMP
using DataFrames
using CSV
using Distances
using LinearAlgebra
using NearestNeighbors
using Plots

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

function knn_fit_predict(X_train, y_train, X_test, k=5)
    k = max(k,3)
    # Number of test points 
    M = size(X_test)[1]
    
    # Initialize predictions 
    y_pred = Matrix{Float64}(undef,M,2)

    kdtree = KDTree(transpose(X_train))
    
    idxs = knn(kdtree, transpose(X_test), k, true)[1]
    for i = 1:M
        x = cf_predict(cf_fit(X_train[idxs[i],:], y_train[idxs[i],:]), X_test[i,:])
        y_pred[i,1] = min(10,max(x[1],-10))
        y_pred[i,2] = min(10,max(x[2],0))
    end
    return y_pred
end

include(joinpath(pwd(), "data_loader_2020.jl"))
# x = all_data[:, ["Offshore DK2", "Offshore DK1", "Onshore DK2", "Onshore DK1", "production_FC", "forward_RE"]]
# x_rf = all_data[:, ["production_FC", "forward_RE"]]
# x_fm = [production_FC_fm, forward_RE]
joinpath(pwd(), "data_loader_2020.jl")
y_train = Matrix(DataFrame(CSV.File(joinpath(pwd(), "results/2020/optimal_everything_training_data.csv"))))
y_test = Matrix(DataFrame(CSV.File(joinpath(pwd(), "results/2020/optimal_everything.csv"))))

test_points = 50

X = x_rf

x_train = Matrix(X[1:8760,:])
x_test = Matrix(X[8761:8760+test_points,:])

k=10
y_pred_knn = knn_fit_predict(x_train, y_train, x_test, k)
plot(y_pred_knn, labels=["$(k)-kNN forward" "$(k)-kNN hydrogen"])

y_pred_gaussian = gaussian_kernel_fit_predict(x_train, y_train, x_test, 0.1)
plot!(y_pred_gaussian, labels=["Gaussian forward" "Gaussian hydrogen"])
plot!(y_test[1:test_points,:], labels=["Opt. forward" "Opt. hydrogen"])
savefig(joinpath(pwd(), "results/2020/kernel_based_model_predictions.png"))