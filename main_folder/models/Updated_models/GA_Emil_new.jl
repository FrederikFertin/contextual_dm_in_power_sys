
using Gurobi
using JuMP
using DataFrames
using CSV

include(joinpath(pwd(), "data_loader_2020.jl"))

#Changed balancing scheme accoring to new 

function get_initial_plan(training_period_length, bidding_start)
    # period length is the amount of timesteps used for training
    # bidding_start is the timestep for the first bid, it is expected that 24 bids are needed

    if (training_period_length % 24 != 0)
        throw(ErrorException("Training period must be a multiple of 24 hours!"))
    end

    offset = bidding_start - training_period_length
    #print(offset)
    periods = collect(1:training_period_length)
   # print(periods)
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
    @variable(initial_plan, b[t in periods], Bin) # Binary variable indicating if we are deficit_settle (1) or surplus_settle (0)
    @variable(initial_plan, E_settle[t in periods])
    @variable(initial_plan, qF[1:(n_features+1)])
    @variable(initial_plan, qH[1:(n_features+1)])

    @variable(initial_plan, 0 <= hydrogen[t in periods])
    @variable(initial_plan, forward_bid[t in periods])

    #Maximize profit
    @objective(initial_plan, Max,
        sum(
            lambda_F[t+offset] * forward_bid[t]
            + lambda_H * hydrogen[t]
            + lambda_DW[t+offset] * E_DW[t]
            - lambda_UP[t+offset] * E_UP[t]
            for t in periods
        ) 
    )


    #Max capacity
    @constraint(initial_plan, wind_capacity_up[t in periods], forward_bid[t] <= max_wind_capacity)
    @constraint(initial_plan, wind_capacity_dw[t in periods], forward_bid[t] >= -max_elec_capacity)
    @constraint(initial_plan, elec_capacity[t in periods], hydrogen[t] <= max_elec_capacity)

    # Power surplus == POSITIVE, deficit == NEGATIVE
    @constraint(initial_plan, settlement[t in periods], E_real[t+offset] - forward_bid[t] - hydrogen[t] == E_settle[t])
    #=
    @constraint(initial_plan, surplus_settle1[t in periods], E_DW[t] >= E_settle[t])
    @constraint(initial_plan, surplus_settle2[t in periods], E_DW[t] <= E_settle[t] + M * b[t])
    @constraint(initial_plan, surplus_settle3[t in periods], E_DW[t] <= M * (1 - b[t]))

    @constraint(initial_plan, deficit_settle1[t in periods], E_UP[t] >= -E_settle[t])
    @constraint(initial_plan, deficit_settle2[t in periods], E_UP[t] <= -E_settle[t] + M * (1 - b[t]))
    @constraint(initial_plan, deficit_settle3[t in periods], E_UP[t] <= M * (b[t]))
    =#
  
    @constraint(initial_plan,balancing[t in periods], E_DW[t] - E_UP[t] == E_settle[t])
    for day in days
        @constraint(initial_plan, sum(hydrogen[t] for t in day) >= min_production)
        for t in day
            index = mod(t, 24)
            if (index == 0)
                index = 24
            end
            @constraint(initial_plan, forward_bid[t] == sum(qF[i] * x[t+offset, i] for i in 1:n_features_rf) + qF[n_features_rf+1])
            @constraint(initial_plan, hydrogen[t] == sum(qH[i] * x[t+offset, i] for i in 1:n_features_rf) + qH[n_features_rf+1])
        end
    end

    optimize!(initial_plan)

    return value.(qF), value.(qH), value.(forward_bid)
end


print("\n\n")
print("\n---------------------------RF new--------------------------------")
print("\n---------------------------RF new --------------------------------")
print("\n---------------------------RF new--------------------------------")
include(joinpath(pwd(), "data_export.jl"))

print("\n\n")
x = all_data[:, ["production_FC","forward_RE"]]
n_features = size(x)[2]
# # #---------------------------RF--------------------------------
for i in 12:12
    n_months = i
    #training_period = month * n_months
    training_period=8760
    validation_period = year
    test_period = 0
    bidding_start = length(lambda_F) - validation_period - test_period

    qF, qH, emil_bids = get_initial_plan(training_period, bidding_start)

    data = vcat([qF[i] for i in 1:(n_features+1)], [qH[i] for i in 1:(n_features+1)])
    names = vcat(["qF$i" for i in 1:(n_features+1)], ["qH$i" for i in 1:(n_features+1)])

    filename = "2020/comparing_architecture/GENERAL_rfnew_mo$n_months"
    easy_export(data, names, filename,)
end
qF, qH, emil_bids = get_initial_plan(training_period, bidding_start)
training_period
bidding_start