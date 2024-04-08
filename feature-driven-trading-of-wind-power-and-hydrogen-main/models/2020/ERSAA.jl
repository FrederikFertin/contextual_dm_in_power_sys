
using Distributions, StatsFuns, StatsBase, JuMP, Gurobi, GLMNet


#Generate ER-SAA scenarios 

function cf_fit(X, y)
    """ Closed form weighted fit """
    XT = transpose(X)
    return inv(XT * X) * XT * y

end 
function generateERSAAScenarios(outputs::Array{Float64,2},inputs::Array{Float64,2},covariates_obs::Array{Float64,1},coeff_true::Array{Float64,2})
#Residual no Jack knife
numDependentVar::Int64 = size(outputs,2)
numIndependentVar::Int64 = size(inputs,2)
numSamples::Int64 = size(outputs,1)
returns_scen_ERSAA = zeros(Float64,numSamples,numDependentVar)

if(size(outputs,1) != size(inputs,1))
    throw(ErrorException("Number of output samples != number of input samples"))
end


if(numSamples < numIndependentVar)
    throw(ErrorException("Expected number of samples > number of covariates in OLS!"))
end


    coeff_full = cf_fit(inputs,outputs)

    returns_scen_ERSAA = repeat(((covariates_obs')*coeff_full)', outer=[1,numSamples])' + outputs - inputs*coeff_full


return returns_scen_ERSAA, coeff_full
end 



function generateJSAAScenarios(outputs::Array{Float64,2},inputs::Array{Float64,2},covariates_obs::Array{Float64,1},)
    """
    Output = Target
    Input = features
    Covariates = 
    """

	numDependentVar::Int64 = size(outputs,2)
	numIndependentVar::Int64 = size(inputs,2)
	numSamples::Int64 = size(outputs,1)
	returns_scen_JSAA = zeros(Float64,numSamples,numDependentVar)
	returns_scen_JpSAA = zeros(Float64,numSamples,numDependentVar)
		
	if(size(outputs,1) != size(inputs,1))
		throw(ErrorException("Number of output samples != number of input samples"))
	end
		
	if(regressionMethod == "ols")
	
		if(numSamples < numIndependentVar)
			throw(ErrorException("Expected number of samples > number of covariates in OLS!"))
		end
		

		XtX_fact = lufact((inputs')*inputs)
		XtY = (inputs')*outputs
		
		# first, compute the OLS coefficients using all data points
		coeff_full = zeros(Float64,numIndependentVar,numDependentVar)
		for d = 1:numDependentVar
			coeff_full[:,d] = XtX_fact\(XtY[:,d])
		end

		
		# next, compute the leave-one-out coefficients		
		for s = 1:numSamples
			coeff_loo = zeros(Float64,numIndependentVar,numDependentVar)
			
			if(numSamples == numIndependentVar)
				range_loo = union([1:s-1,s+1:numSamples]...)
				for d = 1:numDependentVar
					coeff_loo[:,d] = (inputs[range_loo,:])\(outputs[range_loo,d])
				end
			else
				rhs_pert = -inputs[s,:]*((outputs[s,:])')
				term1 = XtX_fact\(inputs[s,:])
				for d = 1:numDependentVar
					term2 = XtX_fact\(rhs_pert[:,d])
					term3 = coeff_full[:,d] + term2
					
					coeff_loo[:,d] = term3 + ((((inputs[s,:])')*term3)/(1.0 - ((inputs[s,:])')*term1))*term1
				end
			end
			
			returns_scen_JSAA[s,:] = ((covariates_obs')*coeff_full)' + outputs[s,:] - (((inputs[s,:])')*coeff_loo)'
			returns_scen_JpSAA[s,:] = ((covariates_obs')*coeff_loo)' + outputs[s,:] - (((inputs[s,:])')*coeff_loo)'
		end
    end
	
	return returns_scen_JSAA, returns_scen_JpSAA
end

returns_scen_ersaa, coeff_est = generateERSAAScenarios(returns_data,covariate_data,covariate_obs,regressionMethod,coeff_true)
#Solve SAA 

