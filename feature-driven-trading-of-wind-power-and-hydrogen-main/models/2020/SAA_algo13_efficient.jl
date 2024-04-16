
using Gurobi
using JuMP
using DataFrames
using CSV
using Statistics
using Plots
using Random
using Distributions
using StatsBase

include(joinpath(pwd(),"data_loader_2020.jl"))
include(joinpath(pwd(), "data_export.jl"))
include(joinpath(pwd(),"models/2020/functions_SLO.jl"))

n = 365
test_points = 24*n

train_errors = pred_errors[:,"production_FC"][1:8760] ./ nominal_wind_capacity

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
all_forward_bids_year = [all_forward_bids[i] for i in 1:24 for j in 1:365]
all_hydrogen_productions_year = [all_hydrogen_productions[i] for i in 1:24 for j in 1:365]

filename = "SLO/SAA_365_days_0_trimmed_1_weighted"
export_SAA(all_forward_bids_year, all_hydrogen_productions_year, filename)
=#

### ---------- Do weighted SAA iteratively through the test set ------------- ###
#=
sigma = 0.5

for i = axes(sqDists,1)
    println("Day: $i")
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

### ----------- Do ER SAA iteratively through the test set ---------------- ###


x_train_days = transpose(reshape(transpose(x_train), 48, :))
x_test_days = transpose(reshape(transpose(x_test), 48, :))
y_test_days = transpose(reshape(transpose(y_test), 24, :)) ./ nominal_wind_capacity

# Training hours with wind divided into categories based on wind strength
x_train_wind = x_train[:,1]
y_train_high = x_train_wind .> 0.75
y_train_low = x_train_wind .<= 0.25
y_train_mid = .!(y_train_high .| y_train_low)

error_high = train_errors[reshape(y_train_high, 8760)]
error_mid = train_errors[reshape(y_train_mid, 8760)]
error_low = train_errors[reshape(y_train_low, 8760)]

error_dist = truncated(Normal(mean(train_errors), std(train_errors)), findmin(train_errors)[1], findmax(train_errors)[1])
error_high_dist = truncated(Normal(mean(error_high), std(error_high)), findmin(error_high)[1], findmax(error_high)[1])
error_mid_dist = truncated(Normal(mean(error_mid), std(error_mid)), findmin(error_mid)[1], findmax(error_mid)[1])
error_low_dist = truncated(Normal(mean(error_low), std(error_low)), findmin(error_low)[1], findmax(error_low)[1])

# All wind production and their prediction errors are normalized by the nominal wind capacity at this point
up_test_days = reshape(lambda_UP[8761:8760+test_points], 24, :)'
dw_test_days = reshape(lambda_DW[8761:8760+test_points], 24, :)'

#scenarios = 1000 # Number of scenarios to generate for each day; 1 = point prediction, 2 = 2 scenarios, etc.

for scenarios in [2]
    # Define method for generating scenarios by adding errors:
    # 1 = level dependent from dist, 2 = constant from dist, 3 = constant from training errors
    method = "dist_levels"
    #method = "dist"
    #method = "train_errors"
    if scenarios == 1
        method = "point"
    end

    for i = 1:n
        Random.seed!(i*2)
        println("Day: $i")

        # Extract data for day i
        local test_day_wind = x_test_days[i,1:2:48]   # Point prediction of wind production
        local test_day_lambda = x_test_days[i,2:2:48] # Point prediction of forward price
        local test_day_up = up_test_days[i,:]
        local test_day_dw = dw_test_days[i,:]

        # Generate scenarios for wind production
        if scenarios == 1
            local sample_predictions = zeros(scenarios, 24)
            for j in 1:24
                sample_predictions[j] = test_day_wind[j]
            end
            local generated_scenarios = clamp!(sample_predictions, 0, 1)  .* nominal_wind_capacity      # Generate scenarios for wind production
        elseif method == "dist_levels"
            local sample_predictions = zeros(scenarios, 24)
            for j in 1:24
                local wind_level = test_day_wind[j]
                if wind_level > 0.75
                    sample_predictions[:,j] = wind_level .+ rand(error_high_dist, scenarios)
                elseif wind_level <= 0.25
                    sample_predictions[:,j]= wind_level .+ rand(error_low_dist, scenarios)
                else
                    sample_predictions[:,j] = wind_level .+ rand(error_mid_dist, scenarios)
                end
            end
            local generated_scenarios = clamp!(sample_predictions, 0, 1)  .* nominal_wind_capacity      # Generate scenarios for wind production
        elseif method == "dist"
            local sample_predictions = zeros(scenarios, 24)
            for j in 1:24
                sample_predictions[:,j] = test_day_wind[j] .+ rand(error_dist, scenarios)
            end
            local generated_scenarios = clamp!(sample_predictions, 0, 1)  .* nominal_wind_capacity      # Generate scenarios for wind production
        elseif method == "train_errors"
            local sample_predictions = zeros(scenarios, 24)
            for j in 1:24
                sample_predictions[:,j] = test_day_wind[j] .+ rand(train_errors, scenarios)
            end
            local generated_scenarios = clamp!(sample_predictions, 0, 1)  .* nominal_wind_capacity      # Generate scenarios for wind production
        else
            error("Method not recognized")
        end

        local all_forward_bids, all_hydrogen_productions = get_ER_SAA_plan(generated_scenarios,
                                                                            test_day_lambda,
                                                                            test_day_up,
                                                                            test_day_dw,
                                                                            test_day_wind)
        
        # Save results from day i
        if i == 1
            global data1 = [all_forward_bids[t] for t = 1:length(all_forward_bids)]
            global data2 = [all_hydrogen_productions[t] for t = 1:length(all_hydrogen_productions)]
        else
            data1 = vcat(data1, [all_forward_bids[t] for t = 1:length(all_forward_bids)])
            data2 = vcat(data2, [all_hydrogen_productions[t] for t = 1:length(all_hydrogen_productions)])
        end
    end
    filename = "SLO/erSAA/erSAA_$(scenarios)_scenarios_$(method)_method_$(n)_days"
    export_SAA(data1, data2, filename)

    #method = "dist_levels"
    method = "dist"
    #method = "train_errors"
    if scenarios == 1
        method = "point"
    end

    for i = 1:n
        Random.seed!(i)
        println("Day: $i")

        # Extract data for day i
        local test_day_wind = x_test_days[i,1:2:48]   # Point prediction of wind production
        local test_day_lambda = x_test_days[i,2:2:48] # Point prediction of forward price
        local test_day_up = up_test_days[i,:]
        local test_day_dw = dw_test_days[i,:]

        # Generate scenarios for wind production
        if scenarios == 1
            local sample_predictions = zeros(scenarios, 24)
            for j in 1:24
                sample_predictions[j] = test_day_wind[j]
            end
            local generated_scenarios = clamp!(sample_predictions, 0, 1)  .* nominal_wind_capacity      # Generate scenarios for wind production
        elseif method == "dist_levels"
            local sample_predictions = zeros(scenarios, 24)
            for j in 1:24
                local wind_level = test_day_wind[j]
                if wind_level > 0.75
                    sample_predictions[:,j] = wind_level .+ rand(error_high_dist, scenarios)
                elseif wind_level <= 0.25
                    sample_predictions[:,j]= wind_level .+ rand(error_low_dist, scenarios)
                else
                    sample_predictions[:,j] = wind_level .+ rand(error_mid_dist, scenarios)
                end
            end
            local generated_scenarios = clamp!(sample_predictions, 0, 1)  .* nominal_wind_capacity      # Generate scenarios for wind production
        elseif method == "dist"
            local sample_predictions = zeros(scenarios, 24)
            for j in 1:24
                sample_predictions[:,j] = test_day_wind[j] .+ rand(error_dist, scenarios)
            end
            local generated_scenarios = clamp!(sample_predictions, 0, 1)  .* nominal_wind_capacity      # Generate scenarios for wind production
        elseif method == "train_errors"
            local sample_predictions = zeros(scenarios, 24)
            for j in 1:24
                sample_predictions[:,j] = test_day_wind[j] .+ rand(train_errors, scenarios)
            end
            local generated_scenarios = clamp!(sample_predictions, 0, 1)  .* nominal_wind_capacity      # Generate scenarios for wind production
        else
            error("Method not recognized")
        end

        local all_forward_bids, all_hydrogen_productions = get_ER_SAA_plan(generated_scenarios,
                                                                            test_day_lambda,
                                                                            test_day_up,
                                                                            test_day_dw,
                                                                            test_day_wind)
        
        # Save results from day i
        if i == 1
            global data1 = [all_forward_bids[t] for t = 1:length(all_forward_bids)]
            global data2 = [all_hydrogen_productions[t] for t = 1:length(all_hydrogen_productions)]
        else
            data1 = vcat(data1, [all_forward_bids[t] for t = 1:length(all_forward_bids)])
            data2 = vcat(data2, [all_hydrogen_productions[t] for t = 1:length(all_hydrogen_productions)])
        end
    end
    filename = "SLO/erSAA/erSAA_$(scenarios)_scenarios_$(method)_method_$(n)_days"
    export_SAA(data1, data2, filename)

    #method = "dist_levels"
    #method = "dist"
    method = "train_errors"
    if scenarios == 1
        method = "point"
    end

    for i = 1:n
        Random.seed!(i)
        println("Day: $i")

        # Extract data for day i
        local test_day_wind = x_test_days[i,1:2:48]   # Point prediction of wind production
        local test_day_lambda = x_test_days[i,2:2:48] # Point prediction of forward price
        local test_day_up = up_test_days[i,:]
        local test_day_dw = dw_test_days[i,:]

        # Generate scenarios for wind production
        if scenarios == 1
            local sample_predictions = zeros(scenarios, 24)
            for j in 1:24
                sample_predictions[j] = test_day_wind[j]
            end
            local generated_scenarios = clamp!(sample_predictions, 0, 1)  .* nominal_wind_capacity      # Generate scenarios for wind production
        elseif method == "dist_levels"
            local sample_predictions = zeros(scenarios, 24)
            for j in 1:24
                local wind_level = test_day_wind[j]
                if wind_level > 0.75
                    sample_predictions[:,j] = wind_level .+ rand(error_high_dist, scenarios)
                elseif wind_level <= 0.25
                    sample_predictions[:,j]= wind_level .+ rand(error_low_dist, scenarios)
                else
                    sample_predictions[:,j] = wind_level .+ rand(error_mid_dist, scenarios)
                end
            end
            local generated_scenarios = clamp!(sample_predictions, 0, 1)  .* nominal_wind_capacity      # Generate scenarios for wind production
        elseif method == "dist"
            local sample_predictions = zeros(scenarios, 24)
            for j in 1:24
                sample_predictions[:,j] = test_day_wind[j] .+ rand(error_dist, scenarios)
            end
            local generated_scenarios = clamp!(sample_predictions, 0, 1)  .* nominal_wind_capacity      # Generate scenarios for wind production
        elseif method == "train_errors"
            local sample_predictions = zeros(scenarios, 24)
            for j in 1:24
                sample_predictions[:,j] = test_day_wind[j] .+ rand(train_errors, scenarios)
            end
            local generated_scenarios = clamp!(sample_predictions, 0, 1)  .* nominal_wind_capacity      # Generate scenarios for wind production
        else
            error("Method not recognized")
        end

        local all_forward_bids, all_hydrogen_productions = get_ER_SAA_plan(generated_scenarios,
                                                                            test_day_lambda,
                                                                            test_day_up,
                                                                            test_day_dw,
                                                                            test_day_wind)
        
        # Save results from day i
        if i == 1
            global data1 = [all_forward_bids[t] for t = 1:length(all_forward_bids)]
            global data2 = [all_hydrogen_productions[t] for t = 1:length(all_hydrogen_productions)]
        else
            data1 = vcat(data1, [all_forward_bids[t] for t = 1:length(all_forward_bids)])
            data2 = vcat(data2, [all_hydrogen_productions[t] for t = 1:length(all_hydrogen_productions)])
        end
    end
    filename = "SLO/erSAA/erSAA_$(scenarios)_scenarios_$(method)_method_$(n)_days"
    export_SAA(data1, data2, filename)
end