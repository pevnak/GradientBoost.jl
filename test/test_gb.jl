using Test
using GradientBoost
using GradientBoost.GB
using GradientBoost.LossFunctions

struct DummyGradientBoost <: GBAlgorithm; end

struct StubGradientBoost <: GBAlgorithm
  loss_function::LossFunction
  sampling_rate::Float64
  learning_rate::Float64
  num_iterations::Int
end

function GradientBoost.GB.build_base_func(
  gb::StubGradientBoost,
  instances,
  labels,
  prev_func_pred,
  psuedo)

  function pred(instances)
    num_instances = size(instances, 1)
    predictions = Array{Float64}(undef, num_instances)
    for i = 1:num_instances
      predictions[i] = sum(instances[i,:])
    end
    predictions
  end

  model_const = 0.5
  return (instances) -> model_const .* pred(instances)
end

sgb_instances = [
  2 2;
  2 4
]
sgb_labels = [
  1.0;
  3.0
]

@testset "Gradient Boost" begin
  @testset "not implemented functions throw an error" begin
    dgb = DummyGradientBoost()
    emp_mat = fill(1,1,1)
    emp_vec = Array[]
    @test_throws ErrorException build_base_func(
      dgb,
      emp_mat,
      emp_vec,
      emp_vec,
      emp_vec
    )
  end

  @testset "stochastic_gradient_boost works" begin
    # Sanity check
    sgb = StubGradientBoost(LeastSquares(), 1.0, 0.5, 1)
    model = stochastic_gradient_boost(sgb, sgb_instances, sgb_labels)
    @test 1 == 1
  end

  @testset "fit returns model" begin
    sgb = StubGradientBoost(LeastSquares(), 1.0, 0.5, 1)
    model = stochastic_gradient_boost(sgb, sgb_instances, sgb_labels)
    @test model isa GBModel
  end

  @testset "predict works" begin
    expected = [
      3.0
      3.5
    ]
    sgb = StubGradientBoost(LeastSquares(), 1.0, 0.5, 1)
    model = stochastic_gradient_boost(sgb, sgb_instances, sgb_labels)
    predictions = predict(model, sgb_instances)
    @test predictions == expected
  end

  @testset "create_sample_indices works" begin
    instances = [1:5 6:10]
    labels = [1:5]

    sgb = StubGradientBoost(LeastSquares(), 1, 1, 1)
    indices = create_sample_indices(sgb, instances, labels)
    @test length(indices) == 5
    @test length(unique(indices)) == 5
    @test minimum(indices) >= 1
    @test maximum(indices) <= 5

    sgb = StubGradientBoost(LeastSquares(), 0.5, 1, 1)
    indices = create_sample_indices(sgb, instances, labels)
    @test length(indices) == 3
    @test length(unique(indices)) == 3
    @test minimum(indices) >= 1
    @test maximum(indices) <= 5
  end
end
