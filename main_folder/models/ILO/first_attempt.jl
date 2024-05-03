using Gurobi
using JuMP
using Statistics
using Distributions
using LinearAlgebra

include(joinpath(pwd(), "data_loader_2020.jl"))

function get_initial_plan(days::Int64)

    periods = collect(1:24) # Hour of day
    #initial_plan
    scenarios = collect(1:div(days*24,24)) # This is scenarios per day
    price_scen = reshape(lambda_F, 24, :) # Price [hour, scen]
    price_UP = reshape(lambda_UP, 24, :) # UP Price [hour, scen]
    price_DW = reshape(lambda_DW, 24, :) # DW Price [hour, scen]

    wind_real  = reshape(all_data[:,"production_RE"], 24, :) .* nominal_wind_capacity # Real wind [hour, scen]

    # Declare Gurobi model
    initial_plan = Model(Gurobi.Optimizer)
    set_silent(initial_plan)
    
    #---------------- Definition of variables -----------------#
    ### 1st Stage variables ###
    @variable(initial_plan, 0 <= hydrogen_plan[t in periods, s in scenarios]) # First stage
    @variable(initial_plan, forward_bid[t in periods, s in scenarios]) # First stage 
    @variable(initial_plan, qF[1:3])
    @variable(initial_plan, qH[1:3])

    ### 2nd Stage variables ###
    @variable(initial_plan, 0 <= E_DW[t in periods, s in scenarios] <= 2*max_wind_capacity)
    @variable(initial_plan, 0 <= E_UP[t in periods, s in scenarios] <= 2*max_wind_capacity)
    @variable(initial_plan, -max_elec_capacity <= EH_extra[t in periods, s in scenarios] <= max_elec_capacity)
    
    #---------------- Objective function -----------------#
    #Maximize profit
    @objective(initial_plan, Max,
        sum(lambda_H * hydrogen_plan[t,s] +
            price_scen[t,s] * forward_bid[t,s]
            + price_DW[t,s] * E_DW[t,s]
            - price_UP[t,s] * E_UP[t,s]
            + lambda_H * EH_extra[t,s]
            for s in scenarios
            for t in periods
        )
    )

    #---------------- Constraints -----------------#
    # Min daily production of hydrogen
    @constraint(initial_plan, sum(hydrogen_plan) >= min_production)

    for t in periods
        for s in scenarios
            #### First stage ####
            
            # Cannot sell and produce more than max wind capacity:
            @constraint(initial_plan, forward_bid[t,s] <= max_wind_capacity)
            # Cannot buy more than max consumption:
            @constraint(initial_plan, forward_bid[t,s] >= -max_elec_capacity)
            # Cannot produce more than max capacity:
            @constraint(initial_plan, hydrogen_plan[t,s] <= max_elec_capacity)

            #### Second stage ####
            # Power surplus == POSITIVE, deficit == NEGATIVE
            @constraint(initial_plan, wind_real[t,s] - forward_bid[t,s] - hydrogen_plan[t,s] == 
                        E_DW[t,s] + EH_extra[t,s] - E_UP[t,s])

            ## Algorithm 13 equivalent ##
            if t == 1
                # Must not reduce below min production
                @constraint(initial_plan, EH_extra[t,s] >=
                            - (sum(hydrogen_plan) - min_production))
            else
                # Must not reduce below min production - can do if we have produced more than min production earlier
                @constraint(initial_plan, EH_extra[t,s] >=
                            - (sum(hydrogen_plan[tt,s] + EH_extra[tt,s] for tt=1:t-1) +
                            sum(hydrogen_plan[tt,s] for tt=t:24) - min_production) )
            end
            # Cannot produce more than max capacity:
            @constraint(initial_plan, EH_extra[t,s] + hydrogen_plan[t,s] <= max_elec_capacity)
            # Cannot produce less than 0:
            @constraint(initial_plan, EH_extra[t,s] + hydrogen_plan[t,s] >= 0)
            if lambda_H < price_DW[t,s]
                @constraint(initial_plan, EH_extra[t,s] <= 0)
            end

            @constraint(initial_plan, forward_bid[t,s] == sum(qF[i] * x_train_days[s,(t-1)*2+i] for i in 1:2) + qF[3])
            @constraint(initial_plan, hydrogen_plan[t,s] == sum(qH[i] * x_train_days[s,(t-1)*2+i] for i in 1:2) + qH[3])
        end
    end

    optimize!(initial_plan)

    return value.(qF), value.(qH)
end

include(joinpath(pwd(), "data_export.jl"))

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

qF, qH = get_initial_plan(n)
data = vcat([qF[i] for i in 1:(n_features+1)], [qH[i] for i in 1:(n_features+1)])
names = vcat(["qF$i" for i in 1:(n_features+1)], ["qH$i" for i in 1:(n_features+1)])

filename = "2020/comparing_architecture/GENERAL_rf_mo$n_months"
easy_export(data, names, filename,)
