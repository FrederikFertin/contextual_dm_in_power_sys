

import gurobipy as gp
from gurobipy import GRB
import numpy as np
import pyepo
from pyepo.model.grb import optGrbModel
import torch
from torch import nn
from torch.utils.data import DataLoader
from gurobipy import Model, GRB, quicksum


# optimization model
class myModel(optGrbModel):
    def __init__(self, weights):
        self.weights = np.array(weights)
        self.num_item = len(weights[0])
        super().__init__()

    def _getModel(self):

        initial_plan = Model("Gurobi.Optimizer")

# Definition of variables
        E_DW = initial_plan.addVars(periods, name="E_DW", lb=0)
        E_UP = initial_plan.addVars(periods, name="E_UP", lb=0)
        b = initial_plan.addVars(periods, vtype=GRB.BINARY, name="b")
        E_settle = initial_plan.addVars(periods, name="E_settle")
        qF = initial_plan.addVars(range(1, n_features + 2), name="qF")
        qH = initial_plan.addVars(range(1, n_features + 2), name="qH")
        hydrogen = initial_plan.addVars(periods, name="hydrogen", lb=0)
        forward_bid = initial_plan.addVars(periods, name="forward_bid")

# Objective: Maximize profit
        initial_plan.setObjective(
            quicksum(
                lambda_F[t + offset] * forward_bid[t] +
                lambda_H * hydrogen[t] +
                lambda_DW[t + offset] * E_DW[t] -
                lambda_UP[t + offset] * E_UP[t]
                for t in periods
            ), GRB.MAXIMIZE
        )

        # Constraints
        # Max capacity
        for t in periods:
            initial_plan.addConstr(forward_bid[t] <= max_wind_capacity, name=f"wind_capacity_up_{t}")
            initial_plan.addConstr(forward_bid[t] >= -max_elec_capacity, name=f"wind_capacity_dw_{t}")
            initial_plan.addConstr(hydrogen[t] <= max_elec_capacity, name=f"elec_capacity_{t}")

        # Power surplus (POSITIVE), deficit (NEGATIVE)
        for t in periods:
            initial_plan.addConstr(E_real[t + offset] - forward_bid[t] - hydrogen[t] == E_settle[t], name=f"settlement_{t}")

        # Finalizing model setup
        initial_plan.update()


        # ceate a model
        m = gp.Model()
        # varibles
        x = m.addVars(self.num_item, name="x", vtype=GRB.BINARY)
        # model sense
        m.modelSense = GRB.MAXIMIZE
        # constraints
        m.addConstr(gp.quicksum([self.weights[0,i] * x[i] for i in range(self.num_item)]) <= 7)
        m.addConstr(gp.quicksum([self.weights[1,i] * x[i] for i in range(self.num_item)]) <= 8)
        m.addConstr(gp.quicksum([self.weights[2,i] * x[i] for i in range(self.num_item)]) <= 9)
        return m, x


# prediction model
class LinearRegression(nn.Module):

    def __init__(self):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(num_feat, num_item)

    def forward(self, x):
        out = self.linear(x)
        return out


if __name__ == "__main__":
    #Generate data for the optimization model

    print("")