```@raw html
<style>
    table {
        display: table !important;
        margin: 2rem auto !important;
        border-top: 2pt solid rgba(0,0,0,0.2);
        border-bottom: 2pt solid rgba(0,0,0,0.2);
    }

    pre, div {
        margin-top: 1.4rem !important;
        margin-bottom: 1.4rem !important;
    }

    .code-output {
        padding: 0.7rem 0.5rem !important;
    }
</style>

<!-- PlutoStaticHTML.Begin -->
<!--
    # This information is used for caching.
    [PlutoStaticHTML.State]
    input_sha = "a87eef967f608486f49f61ca99d167b432e6bbe66155d836b2d9d2bb6e5e732d"
    julia_version = "1.7.2"
-->

<div class="markdown"><h1>Direct Feedback Alignment</h1>
</div>


<div class="markdown"><h2>Handling Dependencies</h2>
</div>

<pre class='language-julia'><code class='language-julia'>using Pkg</code></pre>


<pre class='language-julia'><code class='language-julia'>Pkg.add(url="https://github.com/Rasmuskh/Bender.jl.git")</code></pre>


<pre class='language-julia'><code class='language-julia'>begin
    using Bender, Flux, MLDatasets, Statistics, CUDA
    using Flux: onehotbatch, onecold, logitcrossentropy, throttle
    using Flux.Data: DataLoader
    using Parameters: @with_kw
    using DataFrames
end</code></pre>



<div class="markdown"><h2>Utilities</h2>
</div>

<pre class='language-julia'><code class='language-julia'>@with_kw mutable struct Args
    η::Float64 = 0.0003#3e-4       # learning rate
    batchsize::Int = 64   # batch size
    epochs::Int = 10        # number of epochs
    device::Function = gpu  # set as gpu, if gpu available
end</code></pre>
<pre id='var-@unpack_Args' class='code-output documenter-example-output'>Args</pre>

<pre class='language-julia'><code class='language-julia'>function getdata(args)
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
end</code></pre>
<pre id='var-getdata' class='code-output documenter-example-output'>getdata (generic function with 1 method)</pre>

<pre class='language-julia'><code class='language-julia'>function evaluate(data_loader, model)
    acc = 0
    l = 0
    numbatches = length(data_loader)
    for (x,y) in data_loader
        acc += sum(onecold(cpu(model(x))) .== onecold(cpu(y)))*1 / size(x,2)
        l += logitcrossentropy(model(x), y)
    end
    return acc/numbatches, l/numbatches
end</code></pre>
<pre id='var-evaluate' class='code-output documenter-example-output'>evaluate (generic function with 1 method)</pre>


<div class="markdown"><h2>Defining the Training Loop</h2>
</div>

<pre class='language-julia'><code class='language-julia'>function train(; kws...)
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
end</code></pre>
<pre id='var-train' class='code-output documenter-example-output'>train (generic function with 1 method)</pre>


<div class="markdown"><h2>Defining the model</h2>
</div>

<pre class='language-julia'><code class='language-julia'>function build_model()
    m = Chain(  GenDense(784=&gt;512, 512=&gt;784, relu; forward=linear_asym_∂x),
                GenDense(512=&gt;256, 256=&gt;512, relu; forward=linear_asym_∂x),
                GenDense(256=&gt;10, 10=&gt;256; forward=linear_asym_∂x)
            )
    return m
end</code></pre>
<pre id='var-build_model' class='code-output documenter-example-output'>build_model (generic function with 1 method)</pre>

<pre class='language-julia'><code class='language-julia'>df = train(epochs=10)</code></pre>
<table>
<tr>
<th>loss_train</th>
<th>loss_test</th>
<th>acc_train</th>
<th>acc_test</th>
</tr>
<tr>
<td>0.266108</td>
<td>0.259388</td>
<td>0.922442</td>
<td>0.92498</td>
</tr>
<tr>
<td>0.181013</td>
<td>0.183927</td>
<td>0.947789</td>
<td>0.946114</td>
</tr>
<tr>
<td>0.133806</td>
<td>0.145075</td>
<td>0.960946</td>
<td>0.954527</td>
</tr>
<tr>
<td>0.101212</td>
<td>0.118843</td>
<td>0.970284</td>
<td>0.963842</td>
</tr>
<tr>
<td>0.0818218</td>
<td>0.105554</td>
<td>0.975337</td>
<td>0.96865</td>
</tr>
<tr>
<td>0.0687234</td>
<td>0.0982712</td>
<td>0.978922</td>
<td>0.971154</td>
</tr>
<tr>
<td>0.0586691</td>
<td>0.0940164</td>
<td>0.981457</td>
<td>0.971955</td>
</tr>
<tr>
<td>0.0500062</td>
<td>0.0915856</td>
<td>0.984458</td>
<td>0.973658</td>
</tr>
<tr>
<td>0.0426535</td>
<td>0.0911335</td>
<td>0.986476</td>
<td>0.973157</td>
</tr>
<tr>
<td>0.0378737</td>
<td>0.0935544</td>
<td>0.987877</td>
<td>0.973458</td>
</tr>
</table>


<!-- PlutoStaticHTML.End -->
```