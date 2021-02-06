# Gradient Boosted Decision Trees
module GBDecisionTree

using DecisionTree
using DataStructures
using Statistics

using GradientBoost.GB
using GradientBoost.LossFunctions

export GBDT,
       build_base_func

# Gradient boosted decision tree algorithm.
struct GBDT <: GBAlgorithm
  loss_function::LossFunction
  sampling_rate::Float64
  learning_rate::Float64
  num_iterations::Int
  tree_options::Dict

  function GBDT(;loss_function=LeastSquares(),
    sampling_rate=0.6, learning_rate=0.1, 
    num_iterations=100, tree_options=Dict())

    default_options = Dict(
      :n_subfeatures       => 0,
      :max_depth           => -1,
      :min_samples_leaf    => 5,
      :min_samples_split   => 2,
      :min_purity_increase => 0.0,
    )
    options = merge(default_options, tree_options)
    new(loss_function, sampling_rate, learning_rate, num_iterations, options)
  end
end

function GB.build_base_func(
  gb::GBDT,
  instances,
  labels,
  prev_func_pred,
  psuedo)

  # Train learner
  max_features = size(instances, 2)
  model = build_tree(
    psuedo, instances, 
    gb.tree_options[:n_subfeatures],
    gb.tree_options[:max_depth],
    gb.tree_options[:min_samples_leaf],
    gb.tree_options[:min_samples_split],
    gb.tree_options[:min_purity_increase],
  )
  psuedo_pred = apply_tree(model, instances)

  # Update regions (leaves)
  # NOTE(svs14): Trees are immutable, 
  #              override leaves by having node-to-val mapping.
  inst_node_index = InstanceNodeIndex(model, instances)
  function val_func(node)
    inst_ind = inst_node_index.n2i[node]

    # If loss function is LeastSquares, we don't need need to change values.
    if typeof(gb.loss_function) <: LeastSquares
      val = node.majority
    else
      val = fit_best_constant(gb.loss_function,
        labels[inst_ind],
        psuedo[inst_ind],
        psuedo_pred[inst_ind],
        prev_func_pred[inst_ind]
      )
    end

    val
  end
  val_type = eltype(prev_func_pred)
  n2v = Dict{Leaf, val_type}()
  update_regions!(n2v, model, val_func)

  # Prediction function
  function pred(instances)
    num_instances = size(instances, 1)
    predictions = [
      n2v[instance_to_node(model, instances[i,:])]
      for i in 1:num_instances
    ]
    predictions
  end

  # Produce function that delegates prediction to model
  return (instances) -> pred(instances)
end

# DT Helper Functions

struct InstanceNodeIndex
  i2n::Vector{Leaf}
  n2i::DefaultDict{Leaf, Vector{Int}}

  function InstanceNodeIndex(tree::Union{Leaf,Node}, instances)
    num_instances = size(instances, 1)
    i2n = Vector{Leaf}(undef, num_instances)
    n2i = DefaultDict{Leaf, Vector{Int}}( () -> Int[])

    for i = 1:num_instances
      node = instance_to_node(tree, instances[i,:])
      i2n[i] = node
      push!(n2i[node], i)
    end

    new(i2n, n2i)
  end
end

# Returns respective node of instance.
function instance_to_node(tree::Node, instance)
  # Code adapted from DecisionTree.jl
  features = instance
  if tree.featval == nothing || features[tree.featid] < tree.featval
    return instance_to_node(tree.left, features)
  else
    return instance_to_node(tree.right, features)
  end
end
function instance_to_node(leaf::Leaf, instance)
  return leaf
end

# Update region by having updated leaf value encoded
# in a leaf-to-value mapping.
function update_regions!(n2v::Dict{Leaf, T}, node::Node, val_func::Function) where {T}
  update_regions!(n2v, node.left, val_func)
  update_regions!(n2v, node.right, val_func)
end
function update_regions!(n2v::Dict{Leaf, T}, leaf::Leaf, val_func::Function) where {T}
  n2v[leaf] = val_func(leaf)
end

# Loss function fits
function fit_best_constant(lf::LeastAbsoluteDeviation,
  labels, psuedo, psuedo_pred, prev_func_pred)

  values = labels .- prev_func_pred
  median(values)
end
function fit_best_constant(lf::BinomialDeviance,
  labels, psuedo, psuedo_pred, prev_func_pred)

  num = sum(psuedo)
  denom = sum((labels .- psuedo) .* (1 .- labels .+ psuedo))
  if denom == 0.0
    return 0.0
  else
    return num / denom
  end
end

end # module
