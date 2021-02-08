# Loss function optimized by Boosting algorithm

Loss functions are defined in `loss.jl`. The loss function is defined as a structure derived from abstract type `LossFunction` on which you have to dispatch in function
```
GradientBoost.Losses.loss(lf::LossFunction, y, ŷ)
```
where first parameter are true labels, the second are prediction. 

Boosting requires 
* *Negative gradient of loss function* with respect to prediction (it is negative, because they do gradient descend but with positive sign). The gradient is calculated generically through Zygote, but you can always overload `GradientBoost.Losses.negative_gradient(lf::LossFunction, y, ŷ)` it manually, if you can for example do it faster, or for some other reason.
* *Optimization of a constant predictor* which is implemented generically using Roots.jl and Zygote. If you want to overload it manually, overload `GradientBoost.Losses.minimizing_scalar(lf::LossFunction, y)`
* *Optimization of a adding a new predictor* which is implemented generically using Roots.jl and Zygote. If you want to overload it manually, overload `GradientBoost.Losses.fit_best_constant(lf::LossFunction, y, psuedo, yₕ, y₀)` where `y` are labels, `psuedo` is the negative gradients (are called pseudolabels), `yₕ` is output of samples on the new to-be added hypothesis, and `y₀` is the output on samples of a weighted ensemble without the new classifier. For some functions this can be optimized exactly and therefore for such functions this makes sense to implement it exactly.

The above means, that if you want to add a new loss function, for example logistic loss function, it is sufficient to do
```
struct LogisticLoss <: LossFunction; end

loss(lf::LogisticLoss, y, ŷ) = mean(softplus.(-y .* ŷ))

```
