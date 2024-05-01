using Gurobi
using JuMP
using DataFrames
using CSV
using Distances
using LinearAlgebra
using NearestNeighbors
using Plots
using LinearRegression
using HiGHS

function get_weights(X, X_u, sigma=0.51)
    """ Get weights for weighted fit """
    W = zeros(size(X,1), size(X_u,1))
    for i = axes(X,1)
        W[i,:] = gaussian_kernel(X[i,:], X_u, sigma)
    end
    return W
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
    y_pred = X * beta
    return y_pred
end

function get_deterministic_plan(bidding_start, Y)

    offset = bidding_start
    periods = collect(1:24)

    #Declare Gurobi model
    deterministic_plan = Model(Gurobi.Optimizer)

    #Definition of variables
    @variable(deterministic_plan, 0 <= hydrogen[t in periods])
    @variable(deterministic_plan, forward_bid[t in periods])

    #Maximize profit
    @objective(deterministic_plan, Max,
        sum(
            lambda_F_fc[t+offset] * forward_bid[t]
            +
            lambda_H * hydrogen[t]
            for t in periods
        )
    )

    #Max capacity
    @constraint(deterministic_plan, wind_capacity_up[t in periods], forward_bid[t] <= max_wind_capacity)
    @constraint(deterministic_plan, wind_capacity_dw[t in periods], forward_bid[t] >= -max_elec_capacity)
    @constraint(deterministic_plan, elec_capacity[t in periods], hydrogen[t] <= max_elec_capacity)

    #Min production
    @constraint(deterministic_plan, sum(hydrogen[t] for t in periods) >= min_production)

    #Based on forecasted production
    @constraint(deterministic_plan, bidding[t in periods], forward_bid[t] + hydrogen[t] == max(0, min(deterministic_forecast[t+offset, 1], max_wind_capacity)))

    optimize!(deterministic_plan)

    print("\n\n\nCheck obj: $(objective_value(deterministic_plan))")
    print("\n\n\nCheck bidding_start: $(bidding_start)")
    print("\n\n\n")

    return value.(forward_bid), value.(hydrogen)
end

function get_SAA_plan(weights::Vector{Float64}, bidding_start::Int64, trim_level::Float64=0.0)

    periods = collect(1:24) # Hour of day 
    scens = length(weights)
    #SAA
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
    @variable(SAA, 0 <= hydrogen_plan[t in periods]) # First stage
    @variable(SAA, forward_bid[t in periods]) # First stage 

    ### 2nd Stage variables ###
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
                ) * weights[s]
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
            @constraint(SAA, wind_real[t,s] - forward_bid[t] - hydrogen_plan[t] == 
                        E_DW[t,s] + EH_extra[t,s] - E_UP[t,s])

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
            if lambda_H < price_DW[t,s]
                @constraint(SAA, EH_extra[t,s] <= 0)
            end
        end
    end

    optimize!(SAA)

    #print("\n\n\nCheck obj: $(objective_value(SAA))")
    #print("\n\n\nCheck bidding_start: $(bidding_start)")
    #print("\n\n\n")

    return value.(forward_bid), value.(hydrogen_plan)
end

function export_SAA(all_forward_bids, all_hydrogen_productions, filename)
    data = [
    all_forward_bids,
    all_hydrogen_productions
    ]
    names = [
        "forward bid",
        "hydrogen production"
    ]

    typed_dataseries = [[data[1][t] for t = 1:length(data[1])], [data[2][t] for t = 1:length(data[2])] ]
    df = createDF(typed_dataseries, names)
    export_dataframe(df, filename)
    print("\n\nExported file: $filename\n\n")
end

function get_ER_SAA_plan(test_scenarios::Matrix{Float64},
    lambdas::Vector{Float64},
    price_UP::Vector{Float64},
    price_DW::Vector{Float64},
    wind_fc::Vector{Float64})

    # Generate ER-SAA scenarios
    scens = size(test_scenarios, 1)
    scenarios = collect(1:scens) # Number of scenarios/days generated
    periods = collect(1:size(test_scenarios, 2)) # Hour of day

    # Compute scenario weights
    errors = (test_scenarios' .- wind_fc .* nominal_wind_capacity)'
    weights = exp.(- errors.^2 ./ 10)
    W = sum(weights)
    
    for s in scenarios
        for t in periods
            weights[s,t] = weights[s,t] / W * 24
        end
    end

    # Declare Gurobi model
    SAA = Model(Gurobi.Optimizer)
    set_silent(SAA)

    #---------------- Definition of variables -----------------#
    ### 1st Stage variables ###
    @variable(SAA, 0 <= hydrogen_plan[t in periods]) # First stage
    @variable(SAA, forward_bid[t in periods]) # First stage 

    ### 2nd Stage variables ###
    @variable(SAA, 0 <= E_DW[t in periods, s in scenarios] <= 2*max_wind_capacity)
    @variable(SAA, 0 <= E_UP[t in periods, s in scenarios] <= 2*max_wind_capacity)
    @variable(SAA, -max_elec_capacity <= EH_extra[t in periods, s in scenarios] <= max_elec_capacity)
    
    #---------------- Objective function -----------------#
    #Maximize profit
    @objective(SAA, Max,
        sum(lambda_H   * hydrogen_plan[t] +
            lambdas[t] * forward_bid[t]   +
            sum(( price_DW[t] * E_DW[t,s]
                - price_UP[t] * E_UP[t,s]
                + lambda_H    * EH_extra[t,s]
                ) * 1/scens # weights[s,t] # 1/scens 
            for s in scenarios)
            for t in periods)
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
            @constraint(SAA,
                        test_scenarios[s, t] - forward_bid[t] - hydrogen_plan[t]
                        ==
                        E_DW[t,s] + EH_extra[t,s] - E_UP[t,s])

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
            if lambda_H < price_DW[t]
                @constraint(SAA, EH_extra[t,s] <= 0)
            end
        end
    end

    optimize!(SAA)

    #print("\n\n\nCheck obj: $(objective_value(SAA))")
    #print("\n\n\nCheck bidding_start: $(bidding_start)")
    #print("\n\n\n")

    return value.(forward_bid), value.(hydrogen_plan)
end