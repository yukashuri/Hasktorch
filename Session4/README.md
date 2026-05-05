#Hands-on tasks#

##1 Build and train an AND gate using a simple perceptron##

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

