
# packages
using StatsBase
using MLJ
using Random
using Random
using CategoricalArrays
using PrettyPrinting
import DataFrames
using LossFunctions
using MLJXGBoostInterface
import CSV

# import data set
data = CSV.File("Data/imputed_data.csv", normalizenames = true) |> DataFrames.DataFrame
X = data[:, 9:62]
y = data[:, 63]
schema(X)
scitype(y)

# available models
models(matching(X, y))

# change the scitype of X and y
X = coerce(X, Count => Continuous)
y = coerce(y, Continuous)

# available models
models(matching(X, y))


# xgb_model
@load XGBoostCount
xgb_model = XGBoostCount()


# machine
xgbm = machine(xgb_model, X, y)

# evaluate
mach = fit!(xgbm)

cv=CV(nfolds=3)
evaluate!(mach,
          resampling=cv, 
          measure=[l1, rms, rmslp1], 
          verbosity=0)

# 
r1 = range(xgb_model, :max_depth, lower=3, upper=10)
r2 = range(xgb_model, :min_child_weight, lower=0, upper=5)
tm = TunedModel(model=xgb_model, tuning=Grid(resolution=10),
                resampling=CV(nfolds = 6), ranges=[r1,r2],
                measure=rms)

mtm = machine(tm, X, y)






