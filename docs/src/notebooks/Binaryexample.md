# Binary Neural Network
## Handling dependencies

```
using Pkg; Pkg.add(url="https://github.com/Rasmuskh/Bender.jl.git")
```

```
begin
	using Bender, Flux, MLDatasets
	using Flux: onehotbatch, onecold, logitcrossentropy, throttle
	using Flux.Data: DataLoader
	using Parameters: @with_kw
	using DataFrames
end
```

## Utilities
```
@with_kw mutable struct Args
    η::Float64 = 0.0003     # learning rate
    batchsize::Int = 64     # batch size
    epochs::Int = 10        # number of epochs
    device::Function = gpu  # set as gpu, if gpu available
end
```

```
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



```
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

## Defining the training loop
```
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
    end

	# Return trianing metrics as a DataFrame
	df = DataFrame([loss_train, loss_test, acc_train, acc_test], 
				   [:loss_train, :loss_test, :acc_train, :acc_test])
	return df
end
```
# Defining the model
We now define a model which uses binary weights and binary activation functions (except for the output layer which we allow to have real valued activations).
```
function build_model()
	m = Chain(  GenDense(784=>128, sign_STE; forward=linear_binary_weights),
                GenDense(128=>64, sign_STE; forward=linear_binary_weights),
                GenDense(64=>10; forward=linear_binary_weights))
    return m
end
```
# Training the model
```
df = train(epochs=10); round.(df, digits=3)
```

|    | loss_train | loss_test | acc_train | acc_test |
|:--:|:----------:|:---------:|:---------:|:--------:|
|  1 | 1.866      | 1.796     | 0.73      | 0.742    |
|  2 | 1.502      | 1.487     | 0.792     | 0.799    |
|  3 | 1.34       | 1.369     | 0.816     | 0.815    |
|  4 | 1.26       | 1.233     | 0.824     | 0.828    |
|  5 | 1.274      | 1.236     | 0.821     | 0.823    |
|  6 | 1.168      | 1.14      | 0.84      | 0.845    |
|  7 | 1.168      | 1.122     | 0.846     | 0.852    |
|  8 | 1.138      | 1.155     | 0.842     | 0.847    |
|  9 | 1.108      | 1.105     | 0.848     | 0.853    |
| 10 | 1.039      | 1.053     | 0.864     | 0.869    |