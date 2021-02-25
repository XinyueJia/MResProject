import Pkg
Pkg.add("StatsBase")

# packages
using CSV
using DataFrames
using MLJ

# import data
rawdata = CSV.File("Data/SB-105-Full.csv")
data = DataFrame(rawdata)

# get a quick overview of the variables it contains
schema(data)
