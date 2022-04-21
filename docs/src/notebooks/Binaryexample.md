# Binary Neural Network
This page contains a short tutorial on how to train a binary neural network. It is loosely adapted from the Flux model zoo's MLP example.

We will train neural network with binary {-1,+1} weights and hidden layer activations. The output layer will have real-valued neurons. binary networks require less storage and memory space and when deployed on edge computing devices they can be more computationally effecient than real-valued networks (not so much on traditional CPUs and GPUs which have much better optimized BLAS and CUDA kernels for Float32 operations). 

To create a network with binary weights we need to specify a forward mapping which applies a binary activation function to the weights, when we create the layers. We will also specify a binary activation function to ba applied to the layers output. The tricky part about training binary neural networks is that a binary function such as `sign(x)` has derivative zero everywhere except at the oprigin where it is undefined. A typical workaround is to only use the sign function during the forward pass and replace it with a differentiable surrogate such as `tanh` or `identity` during the backwards pass. This *hack* actually works quite well and a lot of research on binary neural networks explores variations of this idea. We will use the *straight-through estimator*, which is explained by Courbariaux et al in the [BinaryConnect](https://arxiv.org/abs/1511.00363) paper. During a forward pass it acts as the sign function, but during the backwards pass (i.e. during backprop) it is replaced by the identity function.

In *Bender.jl* there is a stocastic and a deterministic variant of the STE, `stoc_sign_STE`  and `sign_STE`. We will make use of the deterministic version here. In a traditional fully connected layer the output is computed by first computing some preactivation $a=Wx+b$ and then applying an elementwise activation function $z=f\circ a$. You can specify a forward mapping for the computation of $a$ when initializing a GenDense layer, which is how we can make the layer use the straight through estimator. We will use the following mapping (which is also exported by Bender.jl)
```julia
function linear_binary_weights(a, x)
    W, b, = a.weight, a.bias
    return sign_STE.(W)*x .+ b
end
```
Using this forward mapping we define a 3 layer fully connected network.
```julia
function build_model()
	m = Chain(  GenDense( 	784=>300, sign_STE, forward=linear_binary_weights),
                GenDense( 	300=>100, sign_STE; forward=linear_binary_weights), 
                GenDense(  	100=>10; forward=linear_binary_weights))
    return m
end
```

The following sections contain some boilerplate code. You can expand them to show the details.
## Handling dependencies
```@raw html
<details><summary>show details</summary>
```
```julia
using Pkg; Pkg.add(url="https://github.com/Rasmuskh/Bender.jl.git")
```

```julia
begin
	using Bender, Flux, MLDatasets
	using Flux: onehotbatch, onecold, logitcrossentropy, throttle
	using Flux.Data: DataLoader
	using Parameters: @with_kw
	using DataFrames
end
```

```@raw html
</details>
```

## Utilities

```@raw html
<details><summary>show details</summary>
```

```julia
@with_kw mutable struct Args
    η::Float64 = 0.0003     # learning rate
    batchsize::Int = 64     # batch size
    epochs::Int = 10        # number of epochs
    device::Function = gpu  # set as gpu, if gpu available
end
```

```julia
function getdata(args)
    ENV["DATADEPS_ALWAYS_ACCEPT"] = "true"

    # Loading Dataset	
    xtrain, ytrain = MLDatasets.MNIST.traindata(Float32)
    xtest, ytest = MLDatasets.MNIST.testdata(Float32)

    # Reshape Data in order to flatten each image into a vector
    xtrain = Flux.flatten(xtrain)
    xtest = Flux.flatten(xtest)

    # One-hot-encode the labels
    ytrain, ytest = onehotbatch(ytrain, 0:9), onehotbatch(ytest, 0:9)

    # Batching
    train_data = DataLoader((xtrain, ytrain), batchsize=args.batchsize, shuffle=true, partial=false)
    test_data = DataLoader((xtest, ytest), batchsize=args.batchsize, partial=false)

    return train_data, test_data
end
```



```julia
function evaluate(data_loader, model)
    acc = 0
	l = 0
	numbatches = length(data_loader)
    for (x,y) in data_loader
        acc += sum(onecold(cpu(model(x))) .== onecold(cpu(y)))*1 / size(x,2)
		l += logitcrossentropy(model(x), y)
    end
    return acc/numbatches, l/numbatches
end
```

```@raw html
</details>
```

## Defining the training loop

```@raw html
<details><summary>show details</summary>
```

```julia
function train(; kws...)
    # Initializing Model parameters 
    args = Args(; kws...)

	# Create arrays for recording training metrics
	acc_train = zeros(Float32, args.epochs)
	acc_test = zeros(Float32, args.epochs)
	loss_train = zeros(Float32, args.epochs)
	loss_test = zeros(Float32, args.epochs)
	
    # Load Data
    train_data,test_data = getdata(args)

    # Construct model
    m = build_model()
	
    train_data = args.device.(train_data)
    test_data = args.device.(test_data)
    m = args.device(m)
    loss(x,y) = logitcrossentropy(m(x), y)
    
    # Training
    opt = ADAM(args.η)
	for epoch=1:args.epochs	
        Flux.train!(loss, params(m), train_data, opt)
		acc_train[epoch], loss_train[epoch] = evaluate(train_data, m)
		acc_test[epoch], loss_test[epoch] = evaluate(test_data, m)
		# clamp weights to lie in the range [-1,+1]
		for layer in m
			layer.weight .= clamp.(layer.weight, -1, 1)
		end
    end

	# Return trianing metrics as a DataFrame
	df = DataFrame([loss_train, loss_test, acc_train, acc_test], 
				   [:loss_train, :loss_test, :acc_train, :acc_test])
	return df, cpu(m)
end
```

```@raw html
</details>
```

# Training the model
Although we constrained the model to use {-1,+1} weights during inference it still manages to solve the problem fairly well. The train function stores the model's loss and accuracy on the train and test set in a DataFrame.

```
df, m = train(epochs=10); round.(df, digits=3)
```

|    | loss_train | loss_test | acc_train | acc_test |
|:--:|:----------:|:---------:|:---------:|:--------:|
|    |   Float32  |  Float32  |  Float32  |  Float32 |
|  1 | 1.9        | 1.862     | 0.789     | 0.792    |
|  2 | 1.56       | 1.542     | 0.824     | 0.83     |
|  3 | 1.365      | 1.361     | 0.85      | 0.849    |
|  4 | 1.362      | 1.382     | 0.846     | 0.852    |
|  5 | 1.294      | 1.34      | 0.859     | 0.859    |
|  6 | 1.234      | 1.224     | 0.866     | 0.87     |
|  7 | 1.268      | 1.269     | 0.865     | 0.869    |
|  8 | 1.306      | 1.395     | 0.857     | 0.855    |
|  9 | 1.222      | 1.26      | 0.873     | 0.873    |
| 10 | 1.22       | 1.263     | 0.867     | 0.867    |