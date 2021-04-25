import Pkg
Pkg.add("CSV")
Pkg.build("PyCall")
Pkg.add("PyCall")
Pkg.build("PyPlot")
Pkg.add("PyPlot")
using RDatasets
cd("/Users/xinyuejia/Desktop/MResProject")

# packages
using CSV
using DataFrames
using MLJ
using PyPlot

# import data
rawdata = CSV.File("Data/SB-105-Full.csv")
data = DataFrame(rawdata)

# get a quick overview of the variables it contains
schema(data)
ismissing.(data)
ENV["PYTHON"]="/usr/local/bin/python3"