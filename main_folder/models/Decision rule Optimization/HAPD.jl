
using Gurobi
using JuMP
using DataFrames
using CSV

include(joinpath(pwd(), "data_loader_2020.jl"))
include(joinpath(pwd(), "data_export.jl"))
top_domain = 53.32 # 90% quantile


function get_initial_plan(pd, training_period_length, bidding_start)
    # period length is the amount of timesteps used for training
    # bidding_start is the timestep for the first bid, it is expected that 24 bids are needed

    if (training_period_length % 24 != 0)
        throw(ErrorException("Training period must be a multiple of 24 hours!"))
    end

    offset = bidding_start - training_period_length
    periods = collect(1:training_period_length)
    days = []
    n_days = Int(training_period_length / 24)
    for i in collect(1:n_days)
        day_offset = (i - 1) * 24
        push!(days, collect(1+day_offset:24+day_offset))
    end

    #Declare Gurobi model
    initial_plan = Model(Gurobi.Optimizer)

    #Definition of variables
    @variable(initial_plan, 0 <= E_DW[t in periods])
    @variable(initial_plan, 0 <= E_UP[t in periods])
    if pd == true
        @variable(initial_plan, qF[1:(n_features+1), 1:24, 1:3])
        @variable(initial_plan, qH[1:(n_features+1), 1:24, 1:3])
    else
        @variable(initial_plan, qF[1:(n_features+1), 1:24])
        @variable(initial_plan, qH[1:(n_features+1), 1:24])

    @variable(initial_plan, 0 <= hydrogen[t in periods])
    @variable(initial_plan, forward_bid[t in periods])
    @variable(initial_plan, -max_elec_capacity <= EH_extra[t in periods] <= max_elec_capacity)

    #Maximize profit
    @objective(initial_plan, Max,
        sum(
            lambda_F[t+offset] * forward_bid[t]
            + lambda_H * hydrogen[t]
            + lambda_DW[t+offset] * E_DW[t]
            - lambda_UP[t+offset] * E_UP[t]
            + lambda_H * EH_extra[t]
            for t in periods
        ) 
    )


    #Max capacity
    @constraint(initial_plan, wind_capacity_up[t in periods], forward_bid[t] <= max_wind_capacity)
    @constraint(initial_plan, wind_capacity_dw[t in periods], forward_bid[t] >= -max_elec_capacity)
    @constraint(initial_plan, elec_capacity[t in periods], hydrogen[t] <= max_elec_capacity)
    # Cannot produce more than max capacity:
    @constraint(initial_plan, elec_capacity_2[t in periods], EH_extra[t] + hydrogen[t] <= max_elec_capacity)
    # Cannot produce less than 0:
    @constraint(initial_plan, elec_min_capacity_2[t in periods], EH_extra[t] + hydrogen[t] >= 0)

    # Power surplus == POSITIVE, deficit == NEGATIVE
    @constraint(initial_plan, settlement[t in periods], E_real[t+offset] - forward_bid[t] - hydrogen[t]
                    == E_DW[t] - E_UP[t] + EH_extra[t])

    

    for day in days
        @constraint(initial_plan, sum(hydrogen[t] for t in day) >= min_production)
        for t in day
            index = mod(t, 24)
            if (index == 0)
                index = 24
            end
            if index == 1
                # Must not reduce below min production
                @constraint(initial_plan, EH_extra[t] >=
                            - (sum(hydrogen[tt] for tt=t:t+23) - min_production))
            else
                # Must not reduce below min production - can do if we have produced more than min production earlier
                @constraint(initial_plan, EH_extra[t] >=
                - (sum(hydrogen[tt] + EH_extra[tt] for tt=t-(index-1):t-1) +
                sum(hydrogen[tt] for tt=t:t+(24-index)) - min_production) )
            end
            if lambda_H < lambda_DW[t+offset]
                @constraint(initial_plan, EH_extra[t] <= 0)
            end
            
            if pd == true
                if lambda_F[t+offset] < lambda_H
                    @constraint(initial_plan, forward_bid[t] == sum(qF[i, index, 1] * x[t+offset, i] for i in 1:n_features) + qF[n_features+1, index, 1])
                    @constraint(initial_plan, hydrogen[t] == sum(qH[i, index, 1] * x[t+offset, i] for i in 1:n_features) + qH[n_features+1, index, 1])
                elseif lambda_F[t+offset] < top_domain
                    @constraint(initial_plan, forward_bid[t] == sum(qF[i, index, 2] * x[t+offset, i] for i in 1:n_features) + qF[n_features+1, index, 2])
                    @constraint(initial_plan, hydrogen[t] == sum(qH[i, index, 2] * x[t+offset, i] for i in 1:n_features) + qH[n_features+1, index, 2])
                else
                    @constraint(initial_plan, forward_bid[t] == sum(qF[i, index, 3] * x[t+offset, i] for i in 1:n_features) + qF[n_features+1, index, 3])
                    @constraint(initial_plan, hydrogen[t] == sum(qH[i, index, 3] * x[t+offset, i] for i in 1:n_features) + qH[n_features+1, index, 3])
                end
            else
                @constraint(initial_plan, forward_bid[t] == sum(qF[i, index] * x[t+offset, i] for i in 1:n_features) + qF[n_features+1, index])
                @constraint(initial_plan, hydrogen[t] == sum(qH[i, index] * x[t+offset, i] for i in 1:n_features) + qH[n_features+1, index])
        end
    end

    optimize!(initial_plan)

    if pd == true
        return [[value.(qF[i, :, d]) for i in 1:(n_features+1)] for d in 1:3], [[value.(qH[i, :, d]) for i in 1:(n_features+1)] for d in 1:3], value.(E_DW), value.(E_UP)
    else
        return [value.(qF[i, :]) for i in 1:(n_features+1)], [value.(qH[i, :]) for i in 1:(n_features+1)], value.(E_DW), value.(E_UP)
end

print("\n\n")
print("\n---------------------------AF--------------------------------")
print("\n---------------------------AF--------------------------------")
print("\n---------------------------AF--------------------------------")
print("\n\n")
#n_features = size(x)[2]
print("Number of features: $(n_features) - all features")
# # #---------------------------AF--------------------------------
training_period = year
validation_period = 0
test_period = 0
bidding_start = length(lambda_F) - validation_period - test_period

x[:,"forward_RE"] = lambda_F_fc

pd = false

qFs, qHs, e_dw_lin, e_up_lin = get_initial_plan(training_period, bidding_start)

if pd == true
    data = vcat([qFs[d][i] for i in 1:(n_features+1) for d in 1:3], [qHs[d][i] for i in 1:(n_features+1) for d in 1:3])
    names = vcat(["qF$(d)_$i" for i in 1:(n_features+1) for d in 1:3], ["qH$(d)_$i" for i in 1:(n_features+1) for d in 1:3])
else
    data = vcat([qFs[i] for i in 1:(n_features+1)], [qHs[i] for i in 1:(n_features+1)])
    names = vcat(["qF$i" for i in 1:(n_features+1)], ["qH$i" for i in 1:(n_features+1)])

filename = "2020/pricedomains/hourly_full_year_lin"
easy_export(data, names, filename,)


# print("\n\n")
# print("\n---------------------------forecast_model--------------------------------")
# print("\n---------------------------forecast_model--------------------------------")
# print("\n---------------------------forecast_model--------------------------------")
# print("\n\n")
# # # #---------------------------forecast_model--------------------------------
# for i in 1:12
#     n_months = i
#     training_period = month * n_months
#     validation_period = year
#     test_period = 0
#     bidding_start = length(lambda_F) - validation_period - test_period

#     qFs, qHs = get_forecasted_plan(training_period, bidding_start)

#     data = vcat([qFs[d][i] for i in 1:3 for d in 1:3], [qHs[d][i] for i in 1:3 for d in 1:3])
#     names = vcat(["qF$(d)_$i" for i in 1:3 for d in 1:3], ["qH$(d)_$i" for i in 1:3 for d in 1:3])

#     filename = "2020/pricedomains/hourly_forecast_model_PRICEDOMAIN_mo$n_months"
#     easy_export(data, names, filename,)
# end

# print("\n\n")
# print("\n---------------------------RF--------------------------------")
# print("\n---------------------------RF--------------------------------")
# print("\n---------------------------RF--------------------------------")
# print("\n\n")
# x = all_data[:, ["production_FC", "forward_RE"]]
# n_features = size(x)[2]
# # # #---------------------------RF--------------------------------
# for i in 1:12
#     n_months = i
#     training_period = month * n_months
#     validation_period = year
#     test_period = 0
#     bidding_start = length(lambda_F) - validation_period - test_period

#     qFs, qHs = get_initial_plan(training_period, bidding_start)

#     data = vcat([qFs[d][i] for i in 1:(n_features+1) for d in 1:3], [qHs[d][i] for i in 1:(n_features+1) for d in 1:3])
#     names = vcat(["qF$(d)_$i" for i in 1:(n_features+1) for d in 1:3], ["qH$(d)_$i" for i in 1:(n_features+1) for d in 1:3])

#     filename = "2020/pricedomains/hourly_rf_PRICEDOMAIN_mo$n_months"
#     easy_export(data, names, filename,)
# end





