
using Gurobi
using JuMP
using DataFrames
using CSV
using Statistics

include(joinpath(pwd(), "data_loader_2020.jl"))


#function get_SAA_plan(bidding_start)

# offset = bidding_start
periods = collect(1:24) # Hour of day 

#SAA 
scenarios = collect(1:div(8760,24)) # This is scenarios per day
price_scen = reshape(lambda_F, 24, :) # Price [hour, scen]
mean_price = mean(price_scen,dims=2)
price_UP = reshape(lambda_UP, 24, :) # UP Price [hour, scen]
price_DW = reshape(lambda_DW, 24, :) # DW Price [hour, scen]

wind_fore = reshape(all_data[:,"production_FC"],24,:) .* nominal_wind_capacity # Forecasted wind [hour, scen]
wind_real  = reshape(all_data[:,"production_RE"],24,:) .* nominal_wind_capacity # Real wind [hour, scen]

pii =1/length(scenarios)
#Make alternative SAA of where hour is not considered ??, no should be there for hydrogen production 

#Declare Gurobi model
SAA = Model(Gurobi.Optimizer)

#Definition of variables
@variable(SAA, 0 <= hydrogen[t in periods]) # First stage
@variable(SAA, forward_bid[t in periods]) # First stage 

#2nd Stage
@variable(SAA, 0 <= E_DW[t in periods, s in scenarios])
@variable(SAA, 0 <= E_UP[t in periods, s in scenarios])
@variable(SAA, b[t in periods, s in scenarios], Bin) # Binary variable indicating if we are deficit_settle (1) or surplus_settle (0)
@variable(SAA, E_settle[t in periods, s in scenarios])

#rerun 
#Maximize profit
@objective(SAA, Max,
    sum( lambda_H * hydrogen[t] +  mean_price[t] * forward_bid[t]   
        +pii*sum(
        price_DW[t,s] * E_DW[t,s]
        -price_UP[t,s] * E_UP[t,s]
        for s in scenarios
        )
        for t in periods
    )
)

#First stage 
#Max capacity
@constraint(SAA, wind_capacity_up[t in periods], forward_bid[t] <= max_wind_capacity)
@constraint(SAA, wind_capacity_dw[t in periods], forward_bid[t] >= -max_elec_capacity)
@constraint(SAA, elec_capacity[t in periods], hydrogen[t] <= max_elec_capacity)

#Min production
@constraint(SAA,hydro_scen, sum(hydrogen[t] for t in periods) >= min_production)
#Based on forecasted production OLD
#@constraint(SAA, bidding[t in periods], forward_bid[t] + hydrogen[t,s] == max(0, min(deterministic_forecast[t+offset, 1], max_wind_capacity)))
#Based on stochastic production  NEW 
#Forecast 
#@constraint(SAA, bidding[t in periods, s in scenarios], forward_bid[t] + hydrogen[t] <= max(0,min(wind_fore[t,s],1) ))
#Real
@constraint(SAA, bidding[t in periods, s in scenarios], forward_bid[t] + hydrogen[t] <=max_wind_capacity)

# @constraint(SAA, bidding[t in periods, s in scenarios], forward_bid[t] + hydrogen[t,s] == wind_real[t,s] )
####################################----------#####################################################

#                           2nd stage constraints (Balancing)
# Power surplus == POSITIVE, deficit == NEGATIVE

@constraint(SAA, settlement[t in periods, s in scenarios], wind_real[t,s] - forward_bid[t] - hydrogen[t] == E_settle[t,s])

@constraint(SAA, surplus_settle1[t in periods, s in scenarios], E_DW[t,s] >= E_settle[t,s])
@constraint(SAA, surplus_settle2[t in periods, s in scenarios], E_DW[t,s] <= E_settle[t,s] + M * b[t,s])
@constraint(SAA, surplus_settle3[t in periods, s in scenarios], E_DW[t,s] <= M * (1 - b[t,s]))

@constraint(SAA, deficit_settle1[t in periods, s in scenarios], E_UP[t,s] >= -E_settle[t,s])
@constraint(SAA, deficit_settle2[t in periods, s in scenarios], E_UP[t,s] <= -E_settle[t,s] + M * (1 - b[t,s]))
@constraint(SAA, deficit_settle3[t in periods, s in scenarios], E_UP[t,s] <= M * (b[t,s]))

optimize!(SAA)

print("\n\n\nCheck obj: $(objective_value(SAA))")
print("\n\n\nCheck bidding_start: $(bidding_start)")
print("\n\n\n")

#    return value.(forward_bid), value.(hydrogen)
#end


validation_period = year
all_forward_bids = []
all_hydrogen_productions = []


#get_SAA_plan(bidding_start)


#=
for i in 1:(365)
    bidding_start = length(lambda_F) - (validation_period) + (i - 1) * 24 # - 24

    forward_bids, hydrogen_production = get_deterministic_plan(bidding_start)

    for j in 1:24
        push!(all_forward_bids, forward_bids[j])
        push!(all_hydrogen_productions, hydrogen_production[j])
    end
end
=#
print("\n\nCheck first 24 forward bids:")

for i in 1:24
    print("\n$(all_forward_bids[i])")
end

print("\n\nCheck first 24 hydrogen prods:")
for i in 1:24
    print("\n$(all_hydrogen_productions[i])")
end

# # #---------------------------EXPORT RESULTS--------------------------------
include(joinpath(pwd(), "data_export.jl"))

data = [
    all_forward_bids,
    all_hydrogen_productions
]
names = [
    "forward bid",
    "hydrogen production"
]

filename = "2020/stochastic_fore"


easy_export(data, names, filename,)