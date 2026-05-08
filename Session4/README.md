#　Hands-on tasks　#

##　1 Build and train an AND gate using a simple perceptron　##

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

##　2 Build a XOR gate using a multi-layer perceptron　##

###　a　###
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

###　b　###
```
data MLPSpec = MLPSpec
  { feature_counts :: [Int],
    nonlinearitySpec :: Tensor -> Tensor
  }
```
ネットワークの設計図
feature_countsは各層のニューロンの数
nonlinearitySpecは活性化関数

```
data MLP = MLP
  { layers :: [Linear],
    nonlinearity :: Tensor -> Tensor
  }
  deriving (Generic, Parameterized)
```
実際にメモリ上に存在するネットワークの構造
Linear型・・・線形層
      　　　　w * x + b という計算を行い、重みとバイアスを持つ
deriving (Generic, Parameterized)はこのネットワークの中に入っている重みとバイアスを自動的に見つけてくれるためのコード
Haskellは副作用を嫌う言語なため、設計と実体を分けて定義し、設計は純粋な値で表現し、実体はランダムに初期化される

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
設計図(MLPSpec)からランダムな初期値を持った実体(MLP)を生み出すための具体的な組み立て手順
MLPSPec {..}は設計図の中に入っているfeature_contentsとnonlinearitySpecという変数をこの中でそのまま使えるように展開するための指示
mkLayerSizesという関数は、各層のLinearを作るために、入力の数と出力の数のペアを作成する関数。今回は入力層２、隠れ層２、出力層１なので、一層目の入力は２出力は２。二層目の入力は２、出力は１となる。よって完成したリストは[(2,2), (2,1)]
uncurry関数は「二つの引数を別々に受け取る関数」を「タプル一つだけ受け取る関数に変換する関数」　ex (+) (1, 2) -> ❌　　uncurry (+) (1, 2) -> ⭕️
linearsには具体的な値を持った重みとバイアスを持つ層のリストが格納される
sampleでランダムに値が入れられる

```
mlp :: MLP -> Tensor -> Tensor
mlp MLP {..} input = foldl' revApply input $ intersperse nonlinearity $ map linear layers
  where
    revApply x f = f x
```
予測を行う処理の定義
層がいくつあっても自動的にパイプラインを組み立ててデータを流し込めるような仕組みを作っている
map linear layersで「層のデータのリスト」を「計算してくれる関数のリスト」に変換している
linear関数はlinear層のデータを使って、Wx+bの計算をする関数
intersperse関数は、リストの要素と要素の間に、指定したものを挟み込む関数
`[1層目の計算, tanh, 2層目の計算]`
foldl' revApply inputでデータをパイプラインに流し込んでいる
以下のような順番でバケツリレーされている
1. inputを「１層目の計算」に入れる→結果A
2. 結果Aをtanhに入れる→結果B
3. 結果Bを「２層目の計算」に入れる→最終的な予測値

学習コード

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
初期化
`batchSize = 2` 一回の学習で、２個のデータをまとめて処理する
`numIters = 2000` 学習ループを2000回繰り返す

```
  trained <- foldLoop init numIters $ \state i -> do
    input <- randIO' [batchSize, 2] >>= return . (toDType Float) . (gt 0.5)
```
学習ループの開始とデータの自動生成を行っている
`randIO` 0.0~0.1のランダムな少数を、2行2列のテンソルで生成する
ランダムに生成された値が、0.5より小さい場合1返される。0.5より大きい場合0が返される。そして[0,1], [1,1]のようなランダムな入力データを毎回２ペア生成する。

```
    let (y, y') = (tensorXOR input, squeezeAll $ model state input)
        loss = mseLoss y y'
```
`y`は正解。`y'`は予測値
mseLossで誤差を計算する。平均二乗誤差が計算される。

```
    (newState, _) <- runStep state optimizer loss 1e-1
    return newState
```
計算されたlossを減らすためにrunStep関数がネットワークの中にある全ての重みとバイアスの勾配を自動微分で計算し、一気に更新する。

```
  where
    optimizer = GD
    tensorXOR :: Tensor -> Tensor
    tensorXOR t = (1 - (1 - a) * (1 - b)) * (1 - (a * b))
      where
        a = select 1 0 t
        b = select 1 1 t
```
算数のみでXORの正解を作っている。

### c ###
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
`device = Device CUDA 0` モデルを配置する計算環境の指定
`hypParams = MLPHypParams device 2 [(3,Sigmoid),(1,Sigmoid)]`ネットワークの設計図、初期設定を定義している
`MLPHypParams`の引数の意味は、第一引数にモデルを配置する計算環境、第二引数に入力層のサイズ、第三引数に隠れ層と出力層の構成のリスト

`((trainedModel,_),losses) <- mapAccumM [1..iter] (initModel,GD) $ \epoc (model,opt) -> do`
`mapAccumM`関数は状態を更新しながらループを回して、各ステップの結果をリストとして蓄積する
`GD`は最適化手法である勾配降下法
```
let loss = sumTensors $ for trainingData $ \(input,output) ->
                  let y = asTensor'' device output
                      y' = mlpLayer model $ asTensor'' device input
                  in mseLoss y y'
```
ロスの計算
4つの訓練データ全てのズレを足し合わせて、このエポックでの全体の誤差とする。
`u <- update model opt loss 1e-1`計算された全体の誤差を元に誤差逆伝播法を自動で実行する。どの重みをどう変えれば誤差が減るかを計算し、オプティマイザを使ってモデルを少し修正する。

bで解読したMlpXor.hsとの違い
- MlpXor.hsは型を自分で定義しているが、これはTorch.Layer.MLPモジュールをそのまま利用している。
- 学習データをMlpXor.hsは実行時に乱数でテンソルを生成し計算しているのに対し、事前に定義した静的なリストを使用している。
- 実行デバイスがMlpXor.hsはデフォルトのデバイスの対し、GPUを明示的に指定している。

### d ###
ステップ関数
`
nonlinearitySpec = toType Float . flip gt (asTensor (0.0 :: Float))
`
この0以下の値の場合0、0以上の場合1とした場合、微分可能でないので、エラーになってしまった。

`
シグモイド関数
`
nonlinearitySpec = Torch.sigmoid
`
学習ループを2000回にした場合
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
結果は0.5付近になってしまい、学習に失敗している。

ループ回数を5000回にした場合
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
学習に成功した

中間層を15にした場合
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
少し正解に近くなったが、精度は低い。

中間層を50にした場合
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
ほぼ学習できているが、ループ回数を増やした時ほどの精度はない。


中間層を100にした場合
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
学習に失敗している。

バッチサイズを４にした場合
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
学習に失敗した。

一番精度が上がるのは、学習ループ回数を増やすことだとわかった。
