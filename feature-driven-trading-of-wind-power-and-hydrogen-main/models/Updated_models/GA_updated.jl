

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
include(joinpath(pwd(),"models/SLO/functions_SLO.jl"))




function updated_GA_plan(n_scenarios::Int64, bidding_start::Int64)
   
   
    periods = collect(1:24) # Hour of day 
    #SAA
    scens =  n_scenarios
    scenarios = collect(1:div(scens*24,24)) # This is scenarios per day
    price_scen = reshape(lambda_F, 24, :) # Price [hour, scen]
    price_UP = reshape(lambda_UP, 24, :) # UP Price [hour, scen]
    price_DW = reshape(lambda_DW, 24, :) # DW Price [hour, scen]

    wind_real  = reshape(all_data[:,"production_RE"], 24, :) .* nominal_wind_capacity # Real wind [hour, scen]

    #quant_cut_off = quantile(weights, trim_level)
    #print("\n\n\nCheck quant_cut_off: $quant_cut_off")
    #print("\nPercentage trimmed: $(sum(weights .< quant_cut_off)/length(weights)*100)\n\n\n")

    # Declare Gurobi model
    SAA = Model(Gurobi.Optimizer)
    set_silent(SAA)
    

    #---------------- Definition of variables -----------------#
    ### 1st Stage variables ###
    @variable(SAA, 0 <= hydrogen_plan[t in periods, s in scenarios]) # First stage
    @variable(SAA, forward_bid[t in periods, s in scenarios]) # First stage 

    ### 2nd Stage variables ###
    @variable(SAA, E_settle[t in periods, s in scenarios])
    @variable(SAA, 0 <= E_DW[t in periods, s in scenarios] <= max_wind_capacity)
    @variable(SAA, 0 <= E_UP[t in periods, s in scenarios] <= max_wind_capacity)
    
    #Linear line fit 
    @variable(SAA, qF[1:(n_features_rf+1)])
    @variable(SAA, qH[1:(n_features_rf+1)])
    
    #---------------- Objective function -----------------#
    #Maximize profit
    @objective(SAA, Max,
        sum(
            sum((lambda_H * hydrogen_plan[t,s] +
                price_scen[t,s] * forward_bid[t,s]
                + price_DW[t,s] * E_DW[t,s]
                - price_UP[t,s] * E_UP[t,s]
                ) * 1/s
            for s in scenarios
            )
            for t in periods
        )
    )

    #---------------- Constraints -----------------#
    # Min daily production of hydrogen

    @constraint(SAA,[s in scenarios], sum(hydrogen_plan[t,s] for t in periods) >= min_production)
    for t in periods
        
        for s in scenarios
        #### First stage ####

        # Cannot buy more than max consumption:
        @constraint(SAA, forward_bid[t,s] >= -max_elec_capacity)
        # Cannot produce more than max capacity:
        @constraint(SAA, hydrogen_plan[t,s] <= max_elec_capacity)
        # Cannot sell and produce more than max wind capacity:
        @constraint(SAA, forward_bid[t,s] + hydrogen_plan[t,s] <= max_wind_capacity)

      
            #### Second stage ####

            # Power surplus == POSITIVE, deficit == NEGATIVE
            @constraint(SAA, wind_real[t,s] - forward_bid[t,s] - hydrogen_plan[t,s] == E_settle[t,s])
            
            
            @constraint(SAA, E_DW[t,s] - E_UP[t,s] == E_settle[t,s])
            #=
            ## Algorithm 13 equivalent ##
            if t == 1
                # Must not reduce below min production
                @constraint(SAA, EH_extra[t,s] >=
                            - (sum(hydrogen_plan) - min_production))
            else
                # Must not reduce below min production - can do if we have produced more than min production earlier
                @constraint(SAA, EH_extra[t,s] >=
                            - (sum(hydrogen_plan[tt] + EH_extra[tt,s] for tt=1:t-1) +
                            sum(hydrogen_plan[tt] for tt=t:24) - min_production) )
            end
            # Cannot produce more than max capacity:
            @constraint(SAA, EH_extra[t,s] + hydrogen_plan[t] <= max_elec_capacity)
            # Cannot produce less than 0:
            @constraint(SAA, EH_extra[t,s] + hydrogen_plan[t] >= 0)
        
            =#
            @constraint(SAA, forward_bid[t,s] == sum(qF[i] * x_train_days[s, t*i] for i in 1:n_features_rf) + qF[n_features_rf+1])
            @constraint(SAA, hydrogen_plan[t,s] == sum(qH[i] * x_train_days[s, t*i] for i in 1:n_features_rf) + qH[n_features_rf+1])

        end
      
    end

    optimize!(SAA)

    print("\n\n\nCheck obj: $(objective_value(SAA))")
    print("\n\n\nCheck bidding_start: $(bidding_start)")
    print("\n\n\n")

    return value.(qF), value.(qH)
end


validation_period = year
all_forward_bids = []
all_hydrogen_productions = []
n_months = 12
training_period = month * n_months
test_period = 0
bidding_start = length(lambda_F) - validation_period - test_period

#Setting training up 

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
x_train_days = transpose(reshape(transpose(x_train), 24*size(X,2), :))
x_test_days = transpose(reshape(transpose(x_test_standardized), 24*size(X,2), :))


y_train = Matrix(Y[1:8760,:])
y_test = Matrix(Y[8761:8760+test_points,:])


#2nd stage parameters 
up_test_days = reshape(lambda_UP[8761:8760+test_points], 24, :)'
dw_test_days = reshape(lambda_DW[8761:8760+test_points], 24, :)'


#calculate weights for linear regression
qFs, qHs  = updated_GA_plan(365,bidding_start)







# # #---------------------------EXPORT RESULTS-------------------------------- # # #
include(joinpath(pwd(), "data_export.jl"))

n_features = size(x_rf)[2]
filename = "2020/GA_updated_nohydro"
data = vcat([qFs[i] for i in 1:(n_features+1)], [qHs[i] for i in 1:(n_features+1)])
names = vcat(["qF$i" for i in 1:(n_features+1)], ["qH$i" for i in 1:(n_features+1)])
easy_export(data, names, filename,)


