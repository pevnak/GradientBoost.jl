# Loss function optimized by Boosting algorithm

Loss functions are defined in `loss.jl`. The loss function is defined as a structure on which you have to dispatch in function 
```
loss(lf::LeastSquares, y, y_pred)
```

