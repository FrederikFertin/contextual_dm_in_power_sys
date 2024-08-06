using StatsPlots, Distributions, Statistics, Random
include("functions_SLO.jl")
include(joinpath(pwd(), "data_loader_2020.jl"))
include(joinpath(pwd(), "data_export.jl"))

z_opt_train = Matrix(DataFrame(CSV.File(joinpath(pwd(), "results/2020/optimal_everything_training_data.csv"))))
z_opt_test = Matrix(DataFrame(CSV.File(joinpath(pwd(), "results/2020/optimal_everything.csv"))))

n = 365
test_points = 24*n
all_data
x[:, "forward_RE"] = lambda_F_fc
X = x
Y = lambda_F

x_train = Matrix(X[1:8760,:])
x_test = Matrix(X[8761:8760+test_points,:])
y_train = Matrix(Y[1:8760,:])
y_test = Matrix(Y[8761:8760+test_points,:])

function rmse(predictions, targets)
    return sqrt(mean((predictions .- targets).^2))
end

#Normalizing and Standardize fucks the predictions 

beta = cf_fit(x_train, y_train)
y_pred_train = x_train * beta
y_pred = x_test * beta
plot(y_pred, labels=["Linear Price"])
plot!(y_test, labels=["True Price"])
plot!(all_data[8670:17520,"forward_FC"],labels=["Simens prediction"])
display(plot!())
#get_deterministic_plan(0, y_test)
RMSE_homebrew = rmse(y_pred,y_test)
RMSE_siemens = rmse(all_data[8761:8760+test_points,"forward_FC"],y_test)

residuals_price = y_pred_train - y_train
pred_errors_forward_homebrew = all_data[1:8760,"forward_RE"]-y_pred_train

p = histogram(residuals_price, bins=100, label="Residuals")
display(p)
norm_dist = fit(Normal, residuals_price)


p = plot(norm_dist, label="Residuals")
display(p)

#get_SAA_plan(0, y_test, norm_dist, 100)
n_features = size(x_train,2)
# Reshape the data to have 48 columns to represent 2*24 hours in a day
x_train_days = transpose(reshape(transpose(x_train), 24*n_features, :))
x_test_days = transpose(reshape(transpose(x_test), 24*n_features, :))

# Distance from each test point to each training point d_ij = ||x_i - x_j||
# i is test point, j is training point
dists = transpose(pairwise(Euclidean(), transpose(x_train_days), transpose(x_test_days), dims=2))
sqDists = dists.^2


y_train = Matrix(Y[1:8760,:])
y_test = Matrix(Y[8761:8760+test_points,:])

y_test
q25 = quantile(vec(y_train), 0.25)
q50 = quantile(vec(y_train), 0.50)
q75 = quantile(vec(y_train), 0.75)


validation_period = year
all_forward_bids = []
all_hydrogen_productions = []
n_months = 12
training_period = month * n_months
test_period = 0
bidding_start = length(lambda_F) - validation_period - test_period


# Training hours with prices divided into categories based on wind strength
y_test_days = transpose(reshape(transpose(y_test), 24, :)) # Point prediction of forward price

###################### SAA ############################
weights = ones(365)/365
trim_level = 0.0

all_forward_bids, all_hydrogen_productions = get_SAA_plan_prices(y_test_days',weights, bidding_start, )

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

filename = "SLO/SAA_365_days_0_trimmed_1_weighted_prices"
export_SAA(all_forward_bids_year, all_hydrogen_productions_year, filename)


##################### Weighted SAA ############################
#=
sigma = 100
k = 30
kernel = "knn"
for k in [365]
    for i = axes(sqDists,1)
        println("Day: $i")
        local weights = zeros(Float64, size(sqDists,2))
        
        if kernel == "knn"
            dist = dists[i, :]
            near = sortperm(vec(dist))
            for j = 1:k
                weights[near[j]] = 1.0
            end
        elseif kernel == "gaussian"
            for j = axes(sqDists,2)
                weights[j] = exp.(- sqDists[i,j] / sigma)
            end
        end
        weights = weights / sum(weights)
        replace!(weights, NaN=>0.0)

        local all_forward_bids, all_hydrogen_productions = get_SAA_plan_prices(y_test_days',weights, bidding_start)
        
        if i == 1
            global data1 = [all_forward_bids[t] for t = 1:length(all_forward_bids)]
            global data2 = [all_hydrogen_productions[t] for t = 1:length(all_hydrogen_productions)]
        else
            data1 = vcat(data1, [all_forward_bids[t] for t = 1:length(all_forward_bids)])
            data2 = vcat(data2, [all_hydrogen_productions[t] for t = 1:length(all_hydrogen_productions)])
        end
    end

    filename = "SLO/wSAA/wSAA_365_days_prices_kernel_$(kernel)_k_$(k)"
    export_SAA(data1, data2, filename)

end
=#
###################### ERSAA ############################

# Training hours with wind divided into categories based on wind strength
x_train_prices = y_pred_train #Forecasted prices siemens
train_errors = lambda_F[1:8760] - y_pred_train

highest_error = quantile(vec(train_errors), 0.99)
lowest_error = quantile(vec(train_errors), 0.01)

x_train_prices = x_train_prices[train_errors .> lowest_error .&& train_errors .< highest_error]
train_errors = train_errors[train_errors .> lowest_error .&& train_errors .< highest_error]

y_train_high = x_train_prices .> q75
y_train_low = x_train_prices .<= q25
y_train_mid = .!(y_train_high .| y_train_low)

error_high = train_errors[reshape(y_train_high, length(train_errors))]
error_mid = train_errors[reshape(y_train_mid, length(train_errors))]
error_low = train_errors[reshape(y_train_low, length(train_errors))]

error_dist = truncated(Normal(mean(train_errors), std(train_errors)), findmin(train_errors)[1], findmax(train_errors)[1])
error_high_dist = truncated(Normal(mean(error_high), std(error_high)), findmin(error_high)[1], findmax(error_high)[1])
error_mid_dist = truncated(Normal(mean(error_mid), std(error_mid)), findmin(error_mid)[1], findmax(error_mid)[1])
error_low_dist = truncated(Normal(mean(error_low), std(error_low)), findmin(error_low)[1], findmax(error_low)[1])

# remove outliers of train_errors

# All wind production and their prediction errors are normalized by the nominal wind capacity at this point
up_test_days = reshape(lambda_UP[8761:8760+test_points], 24, :)'
dw_test_days = reshape(lambda_DW[8761:8760+test_points], 24, :)'
#test_day_lambda[1]
x_test_days[1,6:6:24*n_features]
#scenarios = 1000 # Number of scenarios to generate for each day; 1 = point prediction, 2 = 2 scenarios, etc.
x_test_days[1,6:6:24*n_features]
for scenarios in [1 2 3 4 5 10 25 50 100 200 365 500]
    # Define method for generating scenarios by adding errors:
    # 1 = level dependent from dist, 2 = constant from dist, 3 = constant from training errors
    methods = ["train_errors", "dist", "dist_levels"]
    
    for method in methods
        if scenarios == 1
            method = "point"
        end
        for i = 1:n
            Random.seed!(i*2)
            println("Day: $i")

            # Extract data for day i
            local test_day_wind = x_test_days[i,5:6:24*n_features]   # Point prediction of wind production
            local test_day_lambda = y_pred[1+(i-1)*24:24+(i-1)*24] # Point prediction of forward price
            local test_day_up = up_test_days[i,:]
            local test_day_dw = dw_test_days[i,:]

            # Generate scenarios for wind production
            local sample_predictions = zeros(scenarios, 24)
            if scenarios == 1
                for j in 1:24
                    sample_predictions[j] = test_day_lambda[j]
                end
            else
                for j in 1:24
                    sample_predictions[1,j] = test_day_lambda[j]
                end
                if method == "dist_levels"
                    for j in 1:24
                        local price_level = test_day_lambda[j]
                        if price_level > q75
                            sample_predictions[2:scenarios,j] = price_level .+ rand(error_high_dist, scenarios-1)
                        elseif price_level <= q25
                            sample_predictions[2:scenarios,j]= price_level .+ rand(error_low_dist, scenarios-1)
                        else
                            sample_predictions[2:scenarios,j] = price_level .+ rand(error_mid_dist, scenarios-1)
                        end
                    end
                elseif method == "dist"
                    for j in 1:24
                        sample_predictions[2:scenarios,j] = test_day_lambda[j] .+ rand(error_dist, scenarios-1)
                    end
                elseif method == "train_errors"
                    for j in 1:24
                        sample_predictions[2:scenarios,j] = test_day_lambda[j] .+ rand(train_errors, scenarios-1)
                    end
                else
                    error("Method not recognized")
                end
            end
            sample_predictions[sample_predictions .< 0] .= 0
            local generated_scenarios = sample_predictions         # Generate scenarios for wind production

            local all_forward_bids, all_hydrogen_productions = get_ER_SAA_plan_prices(generated_scenarios,
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
        filename = "SLO/erSAA/erSAA_$(scenarios)_scenarios_$(method)_method_real_model"
        export_SAA(data1, data2, filename)
    end
end