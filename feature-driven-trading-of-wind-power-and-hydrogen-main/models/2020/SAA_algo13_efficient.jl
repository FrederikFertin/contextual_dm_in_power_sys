
using Gurobi
using JuMP
using DataFrames
using CSV
using Statistics
using Plots

include(joinpath(pwd(),"data_loader_2020.jl"))


function get_SAA_plan(weights::Vector{Float64}, bidding_start::Int64, trim_level::Float64=0.0)

    periods = collect(1:24) # Hour of day 
    scens = 365
    #SAA 
    scenarios = collect(1:div(scens*24,24)) # This is scenarios per day
    price_scen = reshape(lambda_F, 24, :) # Price [hour, scen]
    price_UP = reshape(lambda_UP, 24, :) # UP Price [hour, scen]
    price_DW = reshape(lambda_DW, 24, :) # DW Price [hour, scen]

    wind_real  = reshape(all_data[:,"production_RE"], 24, :) .* nominal_wind_capacity # Real wind [hour, scen]
    """
    plot(mean(wind_real[:,1:100],dims=2), title="Real wind production", xlabel="Hour of day", ylabel="Production [MWh]")
    display(plot!())

    plot(mean(price_scen[:,1:100],dims=2), title="Price", xlabel="Hour of day", ylabel="Price [EUR/MWh]")
    plot!(mean(price_UP[:,1:100],dims=2), label="UP Price")
    plot!(mean(price_DW[:,1:100],dims=2), label="DW Price")
    display(plot!())
    """
    quant_cut_off = quantile(weights, trim_level)
    print("\n\n\nCheck quant_cut_off: $quant_cut_off")
    print("\nPercentage trimmed: $(sum(weights .< quant_cut_off)/length(weights)*100)\n\n\n")

    # Declare Gurobi model
    SAA = Model(Gurobi.Optimizer)
    

    #---------------- Definition of variables -----------------#
    ### 1st Stage variables ###
    @variable(SAA, 0 <= hydrogen_plan[t in periods]) # First stage
    @variable(SAA, forward_bid[t in periods]) # First stage 

    ### 2nd Stage variables ###
    @variable(SAA, E_settle[t in periods, s in scenarios])
    @variable(SAA, 0 <= E_DW[t in periods, s in scenarios] <= 2*max_wind_capacity)
    @variable(SAA, 0 <= E_UP[t in periods, s in scenarios] <= 2*max_wind_capacity)
    @variable(SAA, -max_elec_capacity <= EH_extra[t in periods, s in scenarios] <= max_elec_capacity)
    
    #---------------- Objective function -----------------#
    #Maximize profit
    @objective(SAA, Max,
        sum(
            lambda_H * hydrogen_plan[t] +
            sum((price_scen[t,s] * forward_bid[t]
                + price_DW[t,s] * E_DW[t,s]
                - price_UP[t,s] * E_UP[t,s]
                + lambda_H * EH_extra[t,s]
                ) * 1/scens
            for s in scenarios
            )
            for t in periods
        )
    )

    #---------------- Constraints -----------------#
    # Min daily production of hydrogen
    @constraint(SAA, sum(hydrogen_plan) >= min_production)

    for t in periods
        #### First stage ####

        # Cannot buy more than max consumption:
        @constraint(SAA, forward_bid[t] >= -max_elec_capacity)
        # Cannot produce more than max capacity:
        @constraint(SAA, hydrogen_plan[t] <= max_elec_capacity)
        # Cannot sell and produce more than max wind capacity:
        @constraint(SAA, forward_bid[t] + hydrogen_plan[t] <= max_wind_capacity)

        for s in scenarios
            #### Second stage ####

            # Power surplus == POSITIVE, deficit == NEGATIVE
            @constraint(SAA, wind_real[t,s] - forward_bid[t] - hydrogen_plan[t] == E_settle[t,s])
            
            # We are surplus settling:
            @constraint(SAA, E_DW[t,s] + EH_extra[t,s] - E_UP[t,s] == E_settle[t,s])

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
        end
    end

    optimize!(SAA)

    print("\n\n\nCheck obj: $(objective_value(SAA))")
    print("\n\n\nCheck bidding_start: $(bidding_start)")
    print("\n\n\n")

    return value.(forward_bid), value.(hydrogen_plan)
end


validation_period = year
all_forward_bids = []
all_hydrogen_productions = []
n_months = 12
training_period = month * n_months
test_period = 0
bidding_start = length(lambda_F) - validation_period - test_period

weights = ones(365)/365
trim_level = 0.0

all_forward_bids, all_hydrogen_productions = get_SAA_plan(weights, bidding_start, )


print("\n\nCheck first 24 forward bids:")

for i in 1:24
    print("\n$(all_forward_bids[i])")
end

print("\n\nCheck first 24 hydrogen prods:")
for i in 1:24
    print("\n$(all_hydrogen_productions[i,1])")
end

# # #---------------------------EXPORT RESULTS-------------------------------- # # #
include(joinpath(pwd(), "data_export.jl"))

data = [
    all_forward_bids,
    all_hydrogen_productions
]
names = [
    "forward bid",
    "hydrogen production"
]

filename = "2020/SAA_redone_365_days_0_trimmed_1_weighted"

typed_dataseries = [[data[1][t] for t = 1:length(data[1])], [sum(data[2][t,d] for d=1:size(data[2])[2])/size(data[2])[2] for  t = 1:size(data[2])[1]]]
df = createDF(typed_dataseries, names)
export_dataframe(df, filename)
print("\n\nExported file: $filename\n\n")
