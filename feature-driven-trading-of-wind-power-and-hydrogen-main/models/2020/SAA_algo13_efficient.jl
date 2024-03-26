
using Gurobi
using JuMP
using DataFrames
using CSV
using Statistics
using Plots

include(joinpath(pwd(),"data_loader_2020.jl"))
include(joinpath(pwd(), "data_export.jl"))
include(joinpath(pwd(),"models/2020/functions_SLO.jl"))

n = 5
test_points = 24*n

X = x_rf
Y = E_real

x_train = Matrix(X[1:8760,:])
x_test = Matrix(X[8761:8760+test_points,:])

x_train_days = transpose(reshape(transpose(x_train), 48, :))
x_test_days = transpose(reshape(transpose(x_test), 48, :))

# Distance from each test point to each training point d_ij = ||x_i - x_j||
# i is test point, j is training point
dists = transpose(pairwise(Euclidean(), transpose(x_train_days), transpose(x_test_days), dims=2))
sqDists = dists.^2

y_train = Matrix(Y[1:8760,:])
y_test = Matrix(Y[8761:8760+test_points,:])


validation_period = year
all_forward_bids = []
all_hydrogen_productions = []
n_months = 12
training_period = month * n_months
test_period = 0
bidding_start = length(lambda_F) - validation_period - test_period

### --------- Do simple SAA with one bidding strategy for all days ------------ ###
#=
weights = ones(365)/365
trim_level = 0.0
all_forward_bids, all_hydrogen_productions = get_SAA_plan(weights, bidding_start, )

print("\n\nCheck first 24 forward bids:")
for i in 1:24
    print("\n$(all_forward_bids[i])")
end

print("\n\nCheck first 24 hydrogen prods:")
for i in 1:24
    print("\n$(all_hydrogen_productions[i])")
end

filename = "SLO/SAA_365_days_0_trimmed_1_weighted"
export_SAA(all_forward_bids, all_hydrogen_productions, filename)
=#


### ---------- Do weighted SAA iteratively through the test set ------------- ###
sigma = 10^2
data = []
for i = axes(sqDists,1)
    local weights = ones(Float64, size(sqDists,2))
    for j = axes(sqDists,2)
        weights[j] = exp.(- sqDists[i,j] / sigma)
    end
    weights = weights / sum(weights)
    local trim_level = 0.0

    local all_forward_bids, all_hydrogen_productions = get_SAA_plan(weights, bidding_start, trim_level)
    
    if i == 1
        global data1 = [all_forward_bids[t] for t = 1:length(all_forward_bids)]
        global data2 = [all_hydrogen_productions[t] for t = 1:length(all_hydrogen_productions)]
    else
        data1 = vcat(data1, [all_forward_bids[t] for t = 1:length(all_forward_bids)])
        data2 = vcat(data2, [all_hydrogen_productions[t] for t = 1:length(all_hydrogen_productions)])
    end
end

filename = "SLO/wSAA/wSAA_365_days_0_trimmed_testdays_$(n)_sigma_$(sigma)"
export_SAA(data1, data2, filename)
