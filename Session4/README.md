# Hands-on tasks

## 1 Build and train an AND gate using a simple perceptron

```
Epoch 1 - Total Error: 3.0
Epoch 2 - Total Error: 2.0
Epoch 3 - Total Error: 3.0
Epoch 4 - Total Error: 2.0
Epoch 5 - Total Error: 2.0
Epoch 6 - Total Error: 0.0
Epoch 7 - Total Error: 0.0
Epoch 8 - Total Error: 0.0
Epoch 9 - Total Error: 0.0
Epoch 10 - Total Error: 0.0
Final Weights: Tensor Float [2] [ 1.0000   ,  2.0000   ]
Final Bias: Tensor Float [] -2.0000   

Results:
Input: [1,1] -> Output: 1.0
Input: [1,0] -> Output: 0.0
Input: [0,1] -> Output: 0.0
Input: [0,0] -> Output: 0.0
```

```
import Prelude hiding (abs)          
import Torch hiding (step)
import Control.Monad (foldM, forM_)

trainingData :: [([Int],Int)]
trainingData = [([1,1],1),([1,0],0),([0,1],0),([0,0],0)]

-- if x > 0 then 1 else 0
stepFunc :: Tensor -> Tensor
stepFunc x = toType Float (gt x (asTensor (0.0 :: Float)))       


perceptron ::
	Tensor -> -- x
	Tensor -> -- weights
	Tensor -> -- bias
	Tensor    -- output
perceptron x w b = stepFunc (dot x w + b)

calculateError ::
  Tensor ->
  Tensor ->
  Tensor
calculateError yTrue yPred = yTrue - yPred

main :: IO ()
main = do
  let wInit = asTensor ([0.0, 0.0] :: [Float])
      bInit = asTensor (0.0 :: Float)

  -- 1. 状態の型を ((Tensor, Tensor), Float) に変更し、累積誤差を持たせる
  let trainData :: 
        ((Tensor, Tensor), Float) -> 
        ([Int], Int) ->
        IO ((Tensor, Tensor), Float)
      trainData ((currentW, currentB), currentLoss) (x, yTrue) = do
        let xTensor = asTensor (map fromIntegral x :: [Float])
            yTrueTensor = asTensor (fromIntegral yTrue :: Float)
            
            yPred = perceptron xTensor currentW currentB
            errorTensor = calculateError yTrueTensor yPred
            
            -- 【追加】表示用に、このデータでの誤差の絶対値をFloatとして取り出す
            -- (正解からどれくらい離れていたかを計算)
            stepLoss = asValue (abs errorTensor) :: Float
            
            newW = currentW + errorTensor * xTensor
            newB = currentB + errorTensor
            
        -- 2. 重みと一緒に「これまでの誤差 + 今回の誤差」を返す
        return ((newW, newB), currentLoss + stepLoss)

  let epochs = 10
      trainEpoch :: Int -> (Tensor, Tensor) -> IO (Tensor, Tensor)
      trainEpoch 0 params = return params
      trainEpoch n params = do
          -- 3. foldM の初期状態として、誤差 0.0 をセットしてスタートする
          (updatedParams, epochLoss) <- foldM trainData (params, 0.0 :: Float) trainingData
          
          -- 4. 1エポック終わるごとに誤差を画面に出力する
          let currentEpoch = epochs - n + 1
          putStrLn $ "Epoch " ++ show currentEpoch ++ " - Total Error: " ++ show epochLoss
          
          trainEpoch (n - 1) updatedParams

  -- 学習の実行
  putStrLn "Training started..."
  (finalW, finalB) <- trainEpoch epochs (wInit, bInit)
  
  putStrLn $ "Final Weights: " ++ show finalW
  putStrLn $ "Final Bias: " ++ show finalB

  -- テスト（学習したパラメータで正しく推論できるか確認）
  putStrLn "\nResults:"
  forM_ trainingData $ \(x, expected) -> do
      let xTensor = asTensor (map fromIntegral x :: [Float])
          pred = perceptron xTensor finalW finalB
      putStrLn $ "Input: " ++ show x ++ " -> Output: " ++ show (asValue pred :: Float)
```

##　2 Build a XOR gate using a multi-layer perceptron

###　a
```
Iteration: 100 | Loss: Tensor Float []  0.2293   
Iteration: 200 | Loss: Tensor Float []  0.2749   
Iteration: 300 | Loss: Tensor Float []  0.2973   
Iteration: 400 | Loss: Tensor Float []  0.4914   
Iteration: 500 | Loss: Tensor Float []  0.2632   
Iteration: 600 | Loss: Tensor Float []  0.5806   
Iteration: 700 | Loss: Tensor Float []  0.2858   
Iteration: 800 | Loss: Tensor Float []  0.2407   
Iteration: 900 | Loss: Tensor Float []  0.1505   
Iteration: 1000 | Loss: Tensor Float []  0.1672   
Iteration: 1100 | Loss: Tensor Float []  0.3210   
Iteration: 1200 | Loss: Tensor Float []  6.4214e-2
Iteration: 1300 | Loss: Tensor Float []  6.0019e-3
Iteration: 1400 | Loss: Tensor Float []  2.0706e-5
Iteration: 1500 | Loss: Tensor Float []  5.5731e-8
Iteration: 1600 | Loss: Tensor Float []  1.0015e-8
Iteration: 1700 | Loss: Tensor Float []  3.4085e-11
Iteration: 1800 | Loss: Tensor Float []  1.1543e-11
Iteration: 1900 | Loss: Tensor Float []  2.2737e-13
Iteration: 2000 | Loss: Tensor Float []  1.4211e-14
Final Model:
0, 0 => Tensor Float []  2.3842e-7
0, 1 => Tensor Float []  1.0000   
1, 0 => Tensor Float []  1.0000   
1, 1 => Tensor Float []  0.0000
```

###　b
```
data MLPSpec = MLPSpec
  { feature_counts :: [Int],
    nonlinearitySpec :: Tensor -> Tensor
  }
```
Network architecture specification:
* `feature_counts`: The number of neurons in each layer.
* `nonlinearitySpec`: The activation function.

```
data MLP = MLP
  { layers :: [Linear],
    nonlinearity :: Tensor -> Tensor
  }
  deriving (Generic, Parameterized)
```
The actual network structure instantiated in memory:
* `Linear`: A linear layer. It performs the computation `w * x + b` and holds the weights and biases.
* `deriving (Generic, Parameterized)`: Automatically discovers and registers the parameters (weights and biases) contained within the network.

**Note on Haskell's design philosophy:** 
Because Haskell strongly avoids side effects, the network's specification and its actual instance are separated. The design is defined as pure values, whereas the actual instance is randomly initialized.

```
instance Randomizable MLPSpec MLP where
  sample MLPSpec {..} = do
    let layer_sizes = mkLayerSizes feature_counts
    linears <- mapM sample $ map (uncurry LinearSpec) layer_sizes
    return $ MLP {layers = linears, nonlinearity = nonlinearitySpec}
    where
      mkLayerSizes (a : (b : t)) =
        scanl shift (a, b) t
        where
          shift (a, b) c = (b, c)
```
Defines the concrete steps to instantiate an `MLP` with random initial weights from the `MLPSpec` blueprint:

* **`MLPSpec {..}`**: Unpacks the record fields (`feature_counts` and `nonlinearitySpec`) so they can be accessed directly. (Uses Haskell's RecordWildCards).
* **`mkLayerSizes`**: Generates pairs of input and output sizes for each `Linear` layer. For a network with an input of 2, hidden layer of 2, and output of 1, it produces `[(2, 2), (2, 1)]` (Layer 1: 2 in / 2 out, Layer 2: 2 in / 1 out).
* **`uncurry`**: Converts a function taking two separate arguments into one taking a single tuple.
  * Example: `(+) (1, 2)` -> ❌ 
  * `uncurry (+) (1, 2)` -> ⭕️
* **`sample` & `linears`**: `sample` initializes the parameters randomly, and `linears` stores the resulting list of layers containing the actual weights and biases.

```
mlp :: MLP -> Tensor -> Tensor
mlp MLP {..} input = foldl' revApply input $ intersperse nonlinearity $ map linear layers
  where
    revApply x f = f x
```
Defines the prediction process (forward pass). It dynamically builds a computational pipeline that can feed data through any number of layers:

* **`map linear layers`**: Converts the list of layer parameters into a list of executable functions. (The `linear` function applies the `W * x + b` computation using the layer's data).
* **`intersperse`**: Inserts the activation function between each layer's computation.
  * e.g., `[1st layer computation, tanh, 2nd layer computation]`
* **`foldl' revApply input`**: Feeds the input data through the constructed pipeline.

The data flows sequentially like a bucket brigade (relay):
1. Pass `input` through the 1st layer computation -> **Result A**
2. Pass **Result A** through the `tanh` activation -> **Result B**
3. Pass **Result B** through the 2nd layer computation -> **Final prediction value**

```
batchSize = 2
numIters = 2000
  
  init <-
    sample $
      MLPSpec
        { feature_counts = [2, 2, 1],
          nonlinearitySpec = Torch.tanh
        }
```
Initialization. 
`batchSize = 2` The number of data samples processed together in a single training step.  
`numIters = 2000` The total number of iterations for the training loop.  

```
  trained <- foldLoop init numIters $ \state i -> do
    input <- randIO' [batchSize, 2] >>= return . (toDType Float) . (gt 0.5)
```
**Training loop and dynamic data generation:**
* `randIO`: Generates a `2x2` tensor containing random floats between 0.0 and 1.0.
* **Thresholding**: Converts the random floats into binary values (returns 1 if less than 0.5, and 0 if greater). This dynamically generates 2 pairs of random binary input data (e.g., `[0, 1]`, `[1, 1]`) in every iteration.
```
    let (y, y') = (tensorXOR input, squeezeAll $ model state input)
        loss = mseLoss y y'
```
* `y`: The ground truth (correct answer). 
* `y'`:  The model's prediction. 
* `mseLoss`: Computes the Mean Squared Error (MSE) between the prediction and the ground truth. 
```
    (newState, _) <- runStep state optimizer loss 1e-1
    return newState
```
**Parameter update
To minimize the `loss`, the `runStep` function uses automatic differentiation (backpropagation) to calculate the gradients for all weights and biases, and updates them simultaneously.

```
  where
    optimizer = GD
    tensorXOR :: Tensor -> Tensor
    tensorXOR t = (1 - (1 - a) * (1 - b)) * (1 - (a * b))
      where
        a = select 1 0 t
        b = select 1 1 t
```
Generates the correct XOR answers using only arithmetic operations.

### c
```
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE RecordWildCards #-}

module Main where

import Prelude hiding (tanh) 
import Control.Monad (forM_)        --base
--import Data.List (cycle)          --base
--hasktorch
import Torch.Tensor       (asValue)
import Torch.Functional   (mseLoss)
import Torch.Device       (Device(..),DeviceType(..))
import Torch.NN           (sample)
import Torch.Train        (update,showLoss,sumTensors)
import Torch.Control      (mapAccumM)
import Torch.Optim        (GD(..))
import Torch.Tensor.TensorFactories (asTensor'')
import Torch.Layer.MLP    (MLPHypParams(..),ActName(..),mlpLayer)
import ML.Exp.Chart   (drawLearningCurve) --nlp-tools

trainingData :: [([Float],Float)]
trainingData = take 10 $ cycle [([1,1],0),([1,0],1),([0,1],1),([0,0],0)]

main :: IO()
main = do
  let iter = 1500::Int
      device = Device CUDA 0
      hypParams = MLPHypParams device 2 [(3,Sigmoid),(1,Sigmoid)]
  initModel <- sample hypParams
  ((trainedModel,_),losses) <- mapAccumM [1..iter] (initModel,GD) $ \epoc (model,opt) -> do
    let loss = sumTensors $ for trainingData $ \(input,output) ->
                  let y = asTensor'' device output
                      y' = mlpLayer model $ asTensor'' device input
                  in mseLoss y y'
        lossValue = (asValue loss)::Float 
    showLoss 10 epoc lossValue 
    u <- update model opt loss 1e-1
    return (u, lossValue)
  drawLearningCurve "graph-xor.png" "Learning Curve" [("",reverse losses)]
  forM_ ([[1,1],[1,0],[0,1],[0,0]::[Float]]) $ \input -> do
    putStr $ show $ input
    putStr ": "
    putStrLn $ show ((mlpLayer trainedModel $ asTensor'' device input))
  -- print trainedModel
  where for = flip map
```
* `device = Device CUDA 0`: Specifies the hardware device for the model.
* `hypParams = MLPHypParams device 2 [(3,Sigmoid),(1,Sigmoid)]`: Defines the neural network architecture and its initial settings.
* `MLPHypParams`takes three arguments: 
1. The computing device for the model.
2. The size of the input layer.
3. A list containing the configurations for the hidden and output layers.

`((trainedModel,_),losses) <- mapAccumM [1..iter] (initModel,GD) $ \epoc (model,opt) -> do`
* `mapAccumM`: Iterates through the loop while updating the state, accumulating the results of each step into a list.
* `GD`: Stands for Gradient Descent, the optimization method used.
```
let loss = sumTensors $ for trainingData $ \(input,output) ->
                  let y = asTensor'' device output
                      y' = mlpLayer model $ asTensor'' device input
                  in mseLoss y y'
```
### Loss Calculation
Aggregates the errors from all 4 training data samples to determine the total loss for the current epoch.

- **`u <- update model opt loss 1e-1`**: Automatically executes backpropagation based on the total loss. It calculates the necessary weight adjustments to minimize the error and updates the model's parameters using the optimizer.

### Differences from `MlpXor.hs` (decoded in part b)
- **Module Usage**: While `MlpXor.hs` defines its own types, this implementation directly leverages the `Torch.Layer.MLP` module.
- **Training Data**: `MlpXor.hs` generates random tensors for training data at runtime, whereas this script uses a predefined static list.
- **Execution Device**: `MlpXor.hs` relies on the default device, but this explicitly specifies the GPU.

### d
### Step Function
`
nonlinearitySpec = toType Float . flip gt (asTensor (0.0 :: Float))
`
Attempting to use this step function—which outputs `0` for values `<= 0` and `1` for values `> 0`—resulted in a runtime error. This is because a step function is non-differentiable, meaning the optimizer cannot calculate the gradients required for backpropagation.

`
### Sifmoid Function
`
nonlinearitySpec = Torch.sigmoid
`
### `numIters = 2000`
`
Iteration: 1100 | Loss: Tensor Float []  0.2543   
Iteration: 1200 | Loss: Tensor Float []  0.3510   
Iteration: 1300 | Loss: Tensor Float []  0.3135   
Iteration: 1400 | Loss: Tensor Float []  0.2706   
Iteration: 1500 | Loss: Tensor Float []  0.2521   
Iteration: 1600 | Loss: Tensor Float []  0.2412   
Iteration: 1700 | Loss: Tensor Float []  0.2418   
Iteration: 1800 | Loss: Tensor Float []  0.2650   
Iteration: 1900 | Loss: Tensor Float []  0.2615   
Iteration: 2000 | Loss: Tensor Float []  0.2575   
Final Model:
0, 0 => Tensor Float []  0.5310   
0, 1 => Tensor Float []  0.5411   
1, 0 => Tensor Float []  0.5383   
1, 1 => Tensor Float []  0.5491   
`
All output values converged to approximately **0.5**.

### `numInters = 5000`
`
Iteration: 100 | Loss: Tensor Float []  0.2487   
Iteration: 200 | Loss: Tensor Float []  0.2411   
Iteration: 300 | Loss: Tensor Float []  0.3080   
Iteration: 400 | Loss: Tensor Float []  0.2597   
Iteration: 500 | Loss: Tensor Float []  0.3394   
Iteration: 600 | Loss: Tensor Float []  0.2731   
Iteration: 700 | Loss: Tensor Float []  0.2316   
Iteration: 800 | Loss: Tensor Float []  0.1673   
Iteration: 900 | Loss: Tensor Float []  0.3735   
Iteration: 1000 | Loss: Tensor Float []  0.2717   
Iteration: 1100 | Loss: Tensor Float []  0.2236   
Iteration: 1200 | Loss: Tensor Float []  0.1835   
Iteration: 1300 | Loss: Tensor Float []  0.2854   
Iteration: 1400 | Loss: Tensor Float []  0.1061   
Iteration: 1500 | Loss: Tensor Float []  0.1164   
Iteration: 1600 | Loss: Tensor Float []  0.3178   
Iteration: 1700 | Loss: Tensor Float []  0.3059   
Iteration: 1800 | Loss: Tensor Float []  0.1194   
Iteration: 1900 | Loss: Tensor Float []  0.1211   
Iteration: 2000 | Loss: Tensor Float []  9.7972e-2
Iteration: 2100 | Loss: Tensor Float []  5.8387e-2
Iteration: 2200 | Loss: Tensor Float []  2.2433e-2
Iteration: 2300 | Loss: Tensor Float []  1.8620e-2
Iteration: 2400 | Loss: Tensor Float []  7.2328e-3
Iteration: 2500 | Loss: Tensor Float []  3.5052e-4
Iteration: 2600 | Loss: Tensor Float []  1.7001e-4
Iteration: 2700 | Loss: Tensor Float []  1.9708e-4
Iteration: 2800 | Loss: Tensor Float []  6.8594e-5
Iteration: 2900 | Loss: Tensor Float []  4.3768e-5
Iteration: 3000 | Loss: Tensor Float []  1.4340e-5
Iteration: 3100 | Loss: Tensor Float []  4.5668e-6
Iteration: 3200 | Loss: Tensor Float []  6.3190e-6
Iteration: 3300 | Loss: Tensor Float []  8.5200e-7
Iteration: 3400 | Loss: Tensor Float []  1.1239e-7
Iteration: 3500 | Loss: Tensor Float []  2.3594e-7
Iteration: 3600 | Loss: Tensor Float []  7.0001e-8
Iteration: 3700 | Loss: Tensor Float []  1.5425e-8
Iteration: 3800 | Loss: Tensor Float []  7.0032e-9
Iteration: 3900 | Loss: Tensor Float []  4.7738e-9
Iteration: 4000 | Loss: Tensor Float []  5.6403e-11
Iteration: 4100 | Loss: Tensor Float []  3.2191e-10
Iteration: 4200 | Loss: Tensor Float []  3.5840e-10
Iteration: 4300 | Loss: Tensor Float []  7.4268e-11
Iteration: 4400 | Loss: Tensor Float []  1.0360e-11
Iteration: 4500 | Loss: Tensor Float []  9.2406e-12
Iteration: 4600 | Loss: Tensor Float []  4.2668e-12
Iteration: 4700 | Loss: Tensor Float []  5.4357e-13
Iteration: 4800 | Loss: Tensor Float []  2.7178e-13
Iteration: 4900 | Loss: Tensor Float []  8.8818e-14
Iteration: 5000 | Loss: Tensor Float []  7.1054e-14
Final Model:
0, 0 => Tensor Float []  1.1921e-7
0, 1 => Tensor Float []  1.0000   
1, 0 => Tensor Float []  1.0000   
1, 1 => Tensor Float []  8.9407e-7
`
The training was successful.

### Adjusting the hidden layer to 15 units.
`
Iteration: 100 | Loss: Tensor Float []  0.2720   
Iteration: 200 | Loss: Tensor Float []  0.2716   
Iteration: 300 | Loss: Tensor Float []  0.2559   
Iteration: 400 | Loss: Tensor Float []  0.3762   
Iteration: 500 | Loss: Tensor Float []  0.3805   
Iteration: 600 | Loss: Tensor Float []  0.1760   
Iteration: 700 | Loss: Tensor Float []  8.0595e-3
Iteration: 800 | Loss: Tensor Float []  0.2655   
Iteration: 900 | Loss: Tensor Float []  2.1321e-2
Iteration: 1000 | Loss: Tensor Float []  0.1711   
Iteration: 1100 | Loss: Tensor Float []  0.2874   
Iteration: 1200 | Loss: Tensor Float []  0.2878   
Iteration: 1300 | Loss: Tensor Float []  0.2867   
Iteration: 1400 | Loss: Tensor Float []  4.1684e-2
Iteration: 1500 | Loss: Tensor Float []  2.5351e-2
Iteration: 1600 | Loss: Tensor Float []  0.1780   
Iteration: 1700 | Loss: Tensor Float []  0.1676   
Iteration: 1800 | Loss: Tensor Float []  0.1325   
Iteration: 1900 | Loss: Tensor Float []  0.1246   
Iteration: 2000 | Loss: Tensor Float []  0.1316   
Final Model:
0, 0 => Tensor Float []  0.1307   
0, 1 => Tensor Float []  0.6598   
1, 0 => Tensor Float []  0.7024   
1, 1 => Tensor Float []  0.4030   
`
The results started to approach the correct values, but the accuracy remains low.

### Adjusting the hidden layer to 50 units.
`
Iteration: 100 | Loss: Tensor Float []  1.8087e-2
Iteration: 200 | Loss: Tensor Float []  0.2380   
Iteration: 300 | Loss: Tensor Float []  0.2100   
Iteration: 400 | Loss: Tensor Float []  0.2320   
Iteration: 500 | Loss: Tensor Float []  0.6138   
Iteration: 600 | Loss: Tensor Float []  0.3480   
Iteration: 700 | Loss: Tensor Float []  0.7223   
Iteration: 800 | Loss: Tensor Float []  0.1726   
Iteration: 900 | Loss: Tensor Float []  0.4628   
Iteration: 1000 | Loss: Tensor Float []  0.1607   
Iteration: 1100 | Loss: Tensor Float []  3.8685e-3
Iteration: 1200 | Loss: Tensor Float []  3.8478e-2
Iteration: 1300 | Loss: Tensor Float []  5.2448e-3
Iteration: 1400 | Loss: Tensor Float []  1.4943e-3
Iteration: 1500 | Loss: Tensor Float []  2.7464e-3
Iteration: 1600 | Loss: Tensor Float []  1.0041e-3
Iteration: 1700 | Loss: Tensor Float []  3.3425e-4
Iteration: 1800 | Loss: Tensor Float []  3.7690e-5
Iteration: 1900 | Loss: Tensor Float []  1.4626e-5
Iteration: 2000 | Loss: Tensor Float []  4.2046e-6
Final Model:
0, 0 => Tensor Float []  3.4666e-3
0, 1 => Tensor Float []  0.9979   
1, 0 => Tensor Float []  0.9997   
1, 1 => Tensor Float []  3.6744e-3
`
With 15 hidden units, the model successfully began to learn the XOR pattern, though it did not reach the same level of precision as the version with a higher iteration count.

### Adjustin the hidden layer to 100 units.
`
Iteration: 100 | Loss: Tensor Float []  5.5643e-2
Iteration: 200 | Loss: Tensor Float []  0.9768   
Iteration: 300 | Loss: Tensor Float []  0.4665   
Iteration: 400 | Loss: Tensor Float []  0.5208   
Iteration: 500 | Loss: Tensor Float []  1.9436e-2
Iteration: 600 | Loss: Tensor Float []  0.1979   
Iteration: 700 | Loss: Tensor Float []  0.1491   
Iteration: 800 | Loss: Tensor Float []  9.8953e-2
Iteration: 900 | Loss: Tensor Float []  7.2808e-4
Iteration: 1000 | Loss: Tensor Float []  0.4758   
Iteration: 1100 | Loss: Tensor Float []  0.1499   
Iteration: 1200 | Loss: Tensor Float []  6.3653e-3
Iteration: 1300 | Loss: Tensor Float []  9.7968e-3
Iteration: 1400 | Loss: Tensor Float []  0.2750   
Iteration: 1500 | Loss: Tensor Float []  0.3101   
Iteration: 1600 | Loss: Tensor Float []  0.2718   
Iteration: 1700 | Loss: Tensor Float []  0.1485   
Iteration: 1800 | Loss: Tensor Float []  0.3724   
Iteration: 1900 | Loss: Tensor Float []  0.2884   
Iteration: 2000 | Loss: Tensor Float []  1.3772e-4
Final Model:
0, 0 => Tensor Float []  6.4905e-2
0, 1 => Tensor Float []  0.9908   
1, 0 => Tensor Float []  0.5223   
1, 1 => Tensor Float []  0.5572 
`
The training failed to converge.

### Using a batch size of 4.
`
Iteration: 100 | Loss: Tensor Float []  0.2382   
Iteration: 200 | Loss: Tensor Float []  0.2762   
Iteration: 300 | Loss: Tensor Float []  0.2486   
Iteration: 400 | Loss: Tensor Float []  0.3807   
Iteration: 500 | Loss: Tensor Float []  0.2106   
Iteration: 600 | Loss: Tensor Float []  0.2485   
Iteration: 700 | Loss: Tensor Float []  0.2257   
Iteration: 800 | Loss: Tensor Float []  0.2188   
Iteration: 900 | Loss: Tensor Float []  0.1795   
Iteration: 1000 | Loss: Tensor Float []  0.2885   
Iteration: 1100 | Loss: Tensor Float []  0.2573   
Iteration: 1200 | Loss: Tensor Float []  0.2153   
Iteration: 1300 | Loss: Tensor Float []  0.1929   
Iteration: 1400 | Loss: Tensor Float []  0.2584   
Iteration: 1500 | Loss: Tensor Float []  0.2432   
Iteration: 1600 | Loss: Tensor Float []  0.2438   
Iteration: 1700 | Loss: Tensor Float []  0.3645   
Iteration: 1800 | Loss: Tensor Float []  0.2509   
Iteration: 1900 | Loss: Tensor Float []  0.2512   
Iteration: 2000 | Loss: Tensor Float []  0.2392   
Final Model:
0, 0 => Tensor Float []  0.6094   
0, 1 => Tensor Float []  0.5739   
1, 0 => Tensor Float []  0.6163   
1, 1 => Tensor Float []  0.5641 
`
The training failed to converge.

Through various experiments, I concluded that **increasing the number of training iterations** is the most effective way to improve the model's accuracy.
