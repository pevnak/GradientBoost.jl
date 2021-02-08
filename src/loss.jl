# Loss functions.
module LossFunctions
using Statistics
# using GradientBoost.Util

export LossFunction,
       loss,
       negative_gradient,
       minimizing_scalar,
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


# Binomial Deviance (Two Classes {0,1})
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
