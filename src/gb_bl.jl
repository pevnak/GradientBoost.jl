# Gradient Boosted Learner
module GBBaseLearner

export GBBL,
       build_base_func,
       learner_fit,
       learner_predict

using GradientBoost.GB
using GradientBoost.LossFunctions
using GradientBoost.Util

# Gradient boosted base learner algorithm.
struct GBBL <: GBAlgorithm
  loss_function::LossFunction
  sampling_rate::Float64
  learning_rate::Float64
  num_iterations::Int
  learner

  function GBBL(learner; loss_function=LeastSquares(),
    sampling_rate=0.8, learning_rate=0.1, 
    num_iterations=100)

    new(loss_function, sampling_rate, learning_rate, num_iterations, learner)
  end
end

function GB.build_base_func(
    gb::GBBL,
    instances,
    labels,
    prev_func_pred,
    psuedo)

  # Train learner
  lf = gb.loss_function
  learner = gb.learner
  model = learner_fit(lf, learner, instances, psuedo)
  psuedo_pred = learner_predict(lf, learner, model, instances)
  model_const =
    fit_best_constant(lf, labels, psuedo, psuedo_pred, prev_func_pred)

  # Produce function that delegates prediction to model
  return (instances) ->
    model_const .* learner_predict(lf, learner, model, instances)
end

  
"""
  function learner_fit(lf::LossFunction, learner, instances, labels)

  Fits base learner with learner instantiated within this function.

  @param lf Loss function (typically, this is not used).
  @param learner Base learner.
  @param instances Instances.
  @param labels Labels.
"""
function learner_fit end


"""
  function learner_predict(lf::LossFunction, learner, model, instances)

  Predicts on base learner.

  @param lf Loss function (typically, this is not used).
  @param learner Base learner.
  @param model Model produced by base learner.
  @param instances Instances.
"""
function learner_predict end

"""
  fit_best_constant(lf::LossFunction, labels, psuedo, yₕ, y₀)

  Find the best multiplier `α` minimizing error of prediction ` y₀ .+ α .* yₕ`.
  the default implementation relies on combination of `Zygote.jl` and `Roots.jl`,
  but for some loss function (Exponential, Quadratic, etc.) and efficient implementation
  exists, and therefore it can be overloaded and provided.
"""
function fit_best_constant(lf::LossFunction, labels, psuedo, yₕ, y₀)
  f(α) = loss(lf, labels, y₀ .+ α .* yₕ)
  ∇f(α) = gradient(α -> f(α), α)[1]
  α₀ = find_zero(∇f,  0.5)
  α₀
end


function fit_best_constant(lf::LeastSquares,
  labels, psuedo, psuedo_pred, prev_func_pred)

  # No refitting required
  1.0
end

function fit_best_constant(lf::LeastAbsoluteDeviation,
  labels, psuedo, psuedo_pred, prev_func_pred)

  weights = abs.(psuedo_pred)
  values = labels .- prev_func_pred

  for i = 1:length(labels)
    if weights[i] != 0.0
      values[i] /= psuedo_pred[i]
    end
  end

  weighted_median(weights, values)
end

end # module
