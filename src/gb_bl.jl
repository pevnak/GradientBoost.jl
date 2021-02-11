# Gradient Boosted Learner
module GBBaseLearner
using ..LossFunctions
using ..GB: to
using TimerOutputs
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
  model = @timeit to "learner_fit" learner_fit(lf, learner, instances, psuedo)
  psuedo_pred =  @timeit to "learner_predict" learner_predict(lf, learner, model, instances)
  model_const = @timeit to "fit_best_constant" fit_best_constant(lf, labels, psuedo, psuedo_pred, prev_func_pred)

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
end # module