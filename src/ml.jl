# Machine learning API for gradient boosting.
module ML

importall GradientBoost.LossFunctions
importall GradientBoost.GB
importall GradientBoost.GBDecisionTree

export GBProblem,
       GBModel,
       fit!,
       predict!,
       GaussianLoss,
       LaplaceLoss,
       BernoulliLoss,
       GBDT


# Gradient boosting problem.
# NOTE(svs14): Might want to find a better name for this.
type GBProblem
  algorithm::GradientBoost
  output::Symbol
  model

  function GBProblem(algorithm, output=:regression)
    new(algorithm, output, nothing)
  end
end

function fit!(gbp::GBProblem, 
  instances::Matrix{Float64}, labels::Vector{Float64})

  # No special processing required.
  gbp.model = fit(gbp.algorithm, instances, labels)
end

function predict!(gbp::GBProblem, 
  instances::Matrix{Float64})

  # Predict with GB algorithm
  predictions = predict(gbp.model, instances)
  
  # Postprocess according to output and loss function
  predictions = postprocess_pred(
    gbp.output, gbp.algorithm.loss_function, predictions
  )

  predictions
end

# Postprocesses predictions according to 
# output and loss function.
function postprocess_pred(
  output::Symbol, lf::LossFunction, predictions::Vector{Float64})

  if output == :class && typeof(lf) <: BernoulliLoss
    return round(logistic(predictions))
  elseif output == :class_prob && typeof(lf) <: BernoulliLoss
    return logistic(predictions)
  elseif output == :regression && !(typeof(lf) <: BernoulliLoss)
    return predictions
  else
    error("Cannot handle $(output) and $(typeof(lf)) together.")
  end
end

# Logistic function.
function logistic(x)
  1 ./ (1 .+ exp(-x))
end

end # module
