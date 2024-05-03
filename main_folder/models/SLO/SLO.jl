include(joinpath(pwd(),"models/2020/functions_SLO.jl"))
include(joinpath(pwd(), "data_loader_2020.jl"))
include(joinpath(pwd(), "data_export.jl"))

z_opt_train = Matrix(DataFrame(CSV.File(joinpath(pwd(), "results/2020/optimal_everything_training_data.csv"))))
z_opt_test = Matrix(DataFrame(CSV.File(joinpath(pwd(), "results/2020/optimal_everything.csv"))))

n = 365
test_points = 24*n

X = x_rf
Y = E_real

x_train = Matrix(X[1:8760,:])
x_test = Matrix(X[8761:8760+test_points,:])



y_train = Matrix(Y[1:8760,:])
y_test = Matrix(Y[8761:8760+test_points,:])

beta = cf_fit(x_train, y_train)
y_pred_train = cf_predict(beta, x_train)
y_pred = cf_predict(beta, x_test)
plot(y_pred, labels=["Linear Wind"])
plot!(y_test, labels=["True Wind"])
display(plot!())
#get_deterministic_plan(0, y_test)

residuals = y_pred_train - y_train

using StatsPlots, Distributions

p = histogram(residuals, bins=100, label="Residuals")
display(p)
norm_dist = fit(Normal, residuals)


p = plot(norm_dist, label="Residuals")
display(p)

#get_SAA_plan(0, y_test, norm_dist, 100)
