
using Gurobi
using JuMP
using DataFrames
using CSV
using Statistics
using Plots

include(joinpath(pwd(),"data_loader_2020.jl"))
include(joinpath(pwd(), "data_export.jl"))
include(joinpath(pwd(),"models/2020/functions_SLO.jl"))

n = 365
test_points = 24*n

X = x_rf
Y = E_real

x_train = Matrix(X[1:8760,:])
x_test = Matrix(X[8761:8760+test_points,:])

# Standardize the data
x_train1 = (x_train[:,1] .- mean(x_train[:,1])) ./ std(x_train[:,1])
x_train2 = (x_train[:,2] .- mean(x_train[:,2])) ./ std(x_train[:,2])
x_train_standardized = hcat(x_train1, x_train2)

x_test1 = (x_test[:,1] .- mean(x_train[:,1])) ./ std(x_train[:,1])
x_test2 = (x_test[:,2] .- mean(x_train[:,2])) ./ std(x_train[:,2])
x_test_standardized = hcat(x_test1, x_test2)

# Reshape the data to have 48 columns to represent 2*24 hours in a day
x_train_days = transpose(reshape(transpose(x_train_standardized), 48, :))
x_test_days = transpose(reshape(transpose(x_test_standardized), 48, :))

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
all_forward_bids_year = [all_forward_bids[i] for i in 1:24 for j in 1:365]
all_hydrogen_productions_year = [all_hydrogen_productions[i] for i in 1:24 for j in 1:365]

filename = "SLO/SAA_365_days_0_trimmed_1_weighted"
export_SAA(all_forward_bids_year, all_hydrogen_productions_year, filename)


### ---------- Do weighted SAA iteratively through the test set ------------- ###
#=
sigma = 0.5
data = []
for i = axes(sqDists,1)
    local weights = ones(Float64, size(sqDists,2))
    for j = axes(sqDists,2)
        weights[j] = exp.(- sqDists[i,j] / sigma)
    end
    weights = weights / sum(weights)
    replace!(weights, NaN=>0.0)
    local trim_level = 0.05

    local all_forward_bids, all_hydrogen_productions = get_SAA_plan(weights, bidding_start, trim_level)
    
    if i == 1
        global data1 = [all_forward_bids[t] for t = 1:length(all_forward_bids)]
        global data2 = [all_hydrogen_productions[t] for t = 1:length(all_hydrogen_productions)]
    else
        data1 = vcat(data1, [all_forward_bids[t] for t = 1:length(all_forward_bids)])
        data2 = vcat(data2, [all_hydrogen_productions[t] for t = 1:length(all_hydrogen_productions)])
    end
end

filename = "SLO/wSAA/wSAA_365_days_$(trim_level)_trimmed_testdays_$(n)_sigma_$(sigma)"
export_SAA(data1, data2, filename)
=#
