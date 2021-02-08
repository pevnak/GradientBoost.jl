using Mill
using Flux
using PrayTools
using NNlib
using Flux.Losses: logitcrossentropy
using GradientBoost
using Statistics
using Roots
using StatsBase
using GradientBoost.GBBaseLearner
using GradientBoost.LossFunctions
import GradientBoost.GBBaseLearner: learner_fit, fit_best_constant, learner_predict
import GradientBoost.LossFunctions: loss, negative_gradient, minimizing_scalar

# This stuff is needed because the library assumes its works with Matrix. It should 
# be converted to nobs
Base.size(x::Mill.AbstractNode, i::Int) = i == 1 ? Mill.nobs(x) : error("i = 2 does not make sense")
Base.getindex(x::Mill.AbstractNode, i, ::Colon) = x[i]


struct LogisticLoss <: LossFunction; end

loss(lf::LogisticLoss, y, ŷ) = mean(softplus.(-y .* ŷ))

function negative_gradient(lf::LogisticLoss, y, ŷ)
  -gradient(ŷ -> loss(lf, y, ŷ), ŷ)[1]
end

"""
  lf --- type of loss function 
  labels --- true labels
  psuedo --- is the negative gradient at the given solution
  psuedo_pred --- is the prediction of the newly added hypothesis
  prev_func_pred --- previous function prediction

"""
function fit_best_constant(lf::LogisticLoss, labels, psuedo, psuedo_pred, prev_func_pred)
  f(α) = loss(lf, labels, prev_func_pred .+ α .* psuedo_pred)
  ∇f(α) = gradient(α -> f(α), α)[1]
  α₀ = find_zero(∇f,  0.6)
  @show (α₀, f(0), f(α₀))
  α₀
end

function minimizing_scalar(lf::LogisticLoss, y)
  mean(y)
end

pmone(y, τ) = 2(y .> τ) .- 1

struct MillLearner end 

function learner_fit(loss, learner::MillLearner, x, y::Vector{Int})
  mb = initbatchprovider(x, y, 50)
  model = reflectinmodel(x[1], d -> Dense(d, 2))
  opt = ADAM()
  ps = Flux.params(model);
  PrayTools.train!((xy...) -> logitcrossentropy(model(xy[1]).data, xy[2]), ps, mb, opt, 1000)  
  model
end

function learner_fit(loss, learner::MillLearner, x, wy::Vector{<:Real})
  dbg = x, wy
  w = StatsBase.Weights(abs.(wy))
  y = Int.(wy .> 0)
  n = min(100, div(sum(y.>0), 2))
  function mb() 
    ii = sample(1:length(y), w, n, replace = false)
    x[ii], Flux.onehotbatch(y[ii], [0,1])
  end
  model = reflectinmodel(x[1], d -> Dense(d, 2))
  opt = ADAM()
  ps = Flux.params(model);
  PrayTools.train!((xy...) -> logitcrossentropy(model(xy[1]).data, xy[2]), ps, mb, opt, 1000)  
  model
end

function learner_predict(loss, learner::MillLearner,  model, x)
  pmone(softmax(model(x).data)[2,:], 0.5)
end

# You might want to overload this one for speed
# function predict(gb_model::GBModel, instances)
#   outputs = gb_model.base_funcs[1](instances)
#   for i = 2:length(gb_model.base_funcs)
#     outputs .+= gb_model.learning_rate .* gb_model.base_funcs[i](instances)
#   end
#   return outputs
# end



gbbl = GBBL(MillLearner();loss_function = LogisticLoss(), num_iterations=10, learning_rate=1)
gbl = GradientBoost.ML.GBLearner(gbbl, :class)
x = ArrayNode(hcat(randn(Float32, 2,100), randn(Float32, 2,100) .+ 3))
y = vcat(fill(-1,100), fill(1,100))

gbl.model = GradientBoost.ML.fit(gbl.algorithm, x, y)

mean(pmone(GradientBoost.ML.predict(gbl.model, x), 0) .!= y)
