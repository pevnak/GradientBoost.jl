# Loss functions.
module LossFunctions
using Statistics
using Roots
using Zygote
using NNlib
# using GradientBoost.Util

export LossFunction,
       loss,
       negative_gradient,
       minimizing_scalar,
       fit_best_constant,
       LeastSquares,
       LeastAbsoluteDeviation,
       BinomialDeviance

# Loss function.
abstract type LossFunction end

"""
  loss(lf::LossFunction, y, ŷ)

  Calculates loss (a scalar value)
  
  @param lf Loss function.
  @param y True response.
  @param ŷ Approximated response.
"""

function loss end

"""
  negative_gradient(lf::LossFunction, y, ŷ)

  Calculates negative gradient of a loss function.
  
  @param lf Loss function.
  @param y True response.
  @param ŷ Approximated response.

  Default function relies on Zygote, but if analytic  solution exists, it can 
  be overrriden using multiple dispatch on first argument.
"""
function negative_gradient(lf::LossFunction, y, ŷ)
  -gradient(ŷ -> loss(lf, y, ŷ), ŷ)[1]
end


"""
  minimizing_scalar(lf::LossFunction, y)

  Finds scalar value c that minimizes loss L(y, c).

  Default function relies on Zygote.jl + Roots.jl, but if analytic solution 
  exists, it can be overrriden using multiple dispatch on  first argument.

  @param lf Loss function.
  @param y True response.
"""
function minimizing_scalar(lf::LossFunction, y)
  ŷ = ones(eltype(y))
  f(α) = loss(lf, y, α .* ŷ)
  ∇f(α) = gradient(α -> f(α), α)[1]
  α₀ = find_zero(∇f,  0.0)
  α₀
end

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

# LeastSquares
struct LeastSquares <: LossFunction; end

function loss(lf::LeastSquares, y, y_pred)
  mean((y .- y_pred) .^ 2.0)
end

function negative_gradient(lf::LeastSquares, y, y_pred)
  y .- y_pred
end

function minimizing_scalar(lf::LeastSquares, y)
  mean(y)
end

function fit_best_constant(lf::LeastSquares,
  labels, psuedo, psuedo_pred, prev_func_pred)

  # No refitting required
  1.0
end



# LeastAbsoluteDeviation
struct LeastAbsoluteDeviation <: LossFunction; end

function loss(lf::LeastAbsoluteDeviation, y, y_pred)
  mean(abs.(y .- y_pred))
end

function negative_gradient(lf::LeastAbsoluteDeviation, y, y_pred)
  sign.(y .- y_pred)
end

function minimizing_scalar(lf::LeastAbsoluteDeviation, y)
  median(y)
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


# Binomial Deviance (Two Classes {0,1}) seems similar to logistic loss
struct BinomialDeviance <: LossFunction; end

function loss(lf::BinomialDeviance, y, y_pred)
  -2.0 .* mean(y .* y_pred .- log.(1.0 .+ exp.(y_pred)))
end

function negative_gradient(lf::BinomialDeviance, y, y_pred)
  y .- 1.0 ./ (1.0 .+ exp.(-y_pred))
end

function minimizing_scalar(lf::BinomialDeviance, y)
  y_sum = sum(y)
  y_length = length(y)
  log(y_sum / (y_length - y_sum))
end




struct LogisticLoss <: LossFunction; end
"""
  loss(lf::LogisticLoss, y, ŷ)

  logistic loss function `softplus.(-y .* ŷ)`
"""
loss(lf::LogisticLoss, y, ŷ) = mean(softplus.(-y .* ŷ))

end # module
