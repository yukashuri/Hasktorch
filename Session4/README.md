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

##2 Build a XOR gate using a multi-layer perceptron##

###a###
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

###b###
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