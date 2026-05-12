```
module Evaluation where

import Torch


-- | 混同行列 (Confusion Matrix) の作成
-- 引数: クラス数, 正解ラベルのリスト, 予測ラベルのリスト
-- 出力: 2次元リスト [[Int]]
confusionMatrix :: Int -> [Int] -> [Int] -> [[Int]]
confusionMatrix numClasses actuals preds =
    let     
        -- 正解と予測をペア (actual, pred) にする 
        pairs = zip actuals preds
        -- クラス i と j のペア (i, j) が何回出現するかを数える関数
        count target = length (filter (== target) pairs)
    in [[count (i, j) | j <- [0..numClasses-1]] | i <- [0..numClasses-1]]


-- | 正解率 (Accuracy) の計算
accuracy :: [Int] -> [Int] -> Double
accuracy actuals preds = 
    let 
        correctPairs = filter (\( a, p) -> a == p) (zip actuals preds)
        correctCount = length correctPairs
        totalCount = length actuals
    in fromIntegral correctCount / fromIntegral totalCount
    

-- | 指定したクラスに対する Precision (適合率)
precision :: Int -> [Int] -> [Int] -> Double
precision targetClass actuals preds = 
    let 
        -- zip で正解と予測をペアにする
        pairs = zip actuals preds
        -- 分母を求める
        predictedCount = length (filter (\( _, p) -> p == targetClass) pairs)
        -- 分子を求める
        truePositiveCount = length (filter (\( a, p) -> a == targetClass && p == targetClass) pairs)
    in if predictedCount == 0 
        then 0 
        else fromIntegral truePositiveCount / fromIntegral predictedCount

-- | 指定したクラスに対する Recall (再現率)
recall :: Int -> [Int] -> [Int] -> Double
recall targetClass actuals preds = 
    let 
        -- zip で正解と予測をペアにする
        pairs = zip actuals preds
        -- 分母を求める
        actualCount = length (filter (\( a, _) -> a == targetClass) pairs)
        -- 分子を求める
        truePositiveCount = length (filter (\( a, p) -> a == targetClass && p == targetClass) pairs)
    in if actualCount == 0 
        then 0 
        else fromIntegral truePositiveCount / fromIntegral actualCount

-- | 指定したクラスに対する F1スコア (調和平均)
f1Score :: Int -> [Int] -> [Int] -> Double
f1Score targetClass actuals preds = 
    -- 2 * (P * R) / (P + R)
    let p = precision targetClass actuals preds
        r = recall targetClass actuals preds
    in if p + r == 0 
        then 0 
        else 2 * p * r / (p + r)

-- | Macro-F1 スコア: 全クラスのF1スコアの単純平均
macroF1 :: Int -> [Int] -> [Int] -> Double
macroF1 numClasses actuals preds = 
    let f1s = [f1Score i actuals preds | i <- [0..numClasses-1]]
        totalF1 = sum f1s
    in if numClasses == 0 
        then 0 
        else totalF1 / fromIntegral numClasses


-- | Weighted-F1 スコア: 各クラスのデータ数を重みとした平均
weightedF1 :: Int -> [Int] -> [Int] -> Double
weightedF1 numClasses actuals preds = 
    let 
        pairs = zip actuals preds
        -- 各クラスのデータ数を求める
        classCounts = [length (filter (\( a, _) -> a == i) pairs) | i <- [0..numClasses-1]]
        -- 各クラスのF1スコアを求める
        f1s = [f1Score i actuals preds | i <- [0..numClasses-1]]
        -- 重み付きF1スコアを計算
        weightedF1s = zipWith (\count f1 -> fromIntegral count * f1) classCounts f1s
        totalWeightedF1 = sum weightedF1s
        totalSamples = sum classCounts
    in if totalSamples == 0 
        then 0 
        else totalWeightedF1 / fromIntegral totalSamples

-- | Micro-F1 スコア: 全体のTP, FP, FNから計算
microF1 :: Int -> [Int] -> [Int] -> Double
microF1 numClasses actuals preds = 
    let 
        pairs = zip actuals preds
        -- 全クラスのTP, FP, FNを合計する
        truePositives = sum [length (filter (\( a, p) -> a == i && p == i) pairs) | i <- [0..numClasses-1]]
        falsePositives = sum [length (filter (\( a, p) -> a /= i && p == i) pairs) | i <- [0..numClasses-1]]
        falseNegatives = sum [length (filter (\( a, p) -> a == i && p /= i) pairs) | i <- [0..numClasses-1]]
        -- Micro-F1を計算
    in if truePositives + falsePositives + falseNegatives == 0 
        then 0 
        else 2 * fromIntegral truePositives / (2 * fromIntegral truePositives + fromIntegral falsePositives + fromIntegral falseNegatives)

-- ビルドを通すため＆テスト用の main 関数
main :: IO ()
main = do
    putStrLn "=== 評価関数のテスト ==="
    
    -- テストデータ（クラス数: 3）
    let actuals = [0, 1, 2, 0, 1, 2, 0, 0]
    let preds   = [0, 2, 2, 0, 1, 1, 0, 2]
    let numClasses = 3
    
    putStrLn "\n[混同行列]"
    let cm = confusionMatrix numClasses actuals preds
    -- print だと横一列で見にくいので、mapM_ print で1行ずつ綺麗に表示させます
    mapM_ print cm

    putStrLn "\n[Accuracy]"
    print (accuracy actuals preds)

    putStrLn "\n[Precision (Class 2)]"
    -- クラス2に対する適合率を計算
    print (precision 2 actuals preds)

    putStrLn "\n[Recall (Class 2)]"
    -- クラス2に対する再現率を計算
    print (recall 2 actuals preds)

    putStrLn "\n[F1 Score (Class 2)]"
    -- クラス2に対するF1スコアを計算
    print (f1Score 2 actuals preds)

    putStrLn "\n[Macro-F1]"
    print (macroF1 numClasses actuals preds)
    putStrLn "\n[Weighted-F1]"
    print (weightedF1 numClasses actuals preds)
    putStrLn "\n[Micro-F1]"
    print (microF1 numClasses actuals preds)

```

```
[混同行列]
[3,0,1]
[0,1,1]
[0,1,1]

[Accuracy]
0.625

[Precision (Class 2)]
0.3333333333333333

[Recall (Class 2)]
0.5

[F1 Score (Class 2)]
0.4

[Macro-F1]
0.5857142857142857

[Weighted-F1]
0.6535714285714286

[Micro-F1]
0.625
```

```
{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE FunctionalDependencies #-}
{-# LANGUAGE RecordWildCards #-}

module Admit where

import Control.Monad (when, foldM) -- foldMを追加
import Data.List (foldl', intersperse, scanl', transpose)
import GHC.Generics
import Torch
import Torch.NN      
import Torch.Optim   
import qualified Data.ByteString.Lazy as BL
import Data.Csv
import qualified Data.Vector as V
import Chart (drawLearningCurve)
--------------------------------------------------------------------------------
-- Data Loading (Kaggle Admission Data)
--------------------------------------------------------------------------------

-- CSVの1行分を表すデータ型
data AdmitData = AdmitData
  { serialNo :: Float
  , gre      :: Float
  , toefl    :: Float
  , rating   :: Float
  , sop      :: Float
  , lor      :: Float
  , cgpa     :: Float
  , research :: Float
  , chance   :: Float
  } deriving (Generic, FromRecord, Show)
  -- FromRecord をつけると、cassavaが自動的にCSVの列と紐づけてくれます


-- CSVファイルを読み込んで、(入力テンソル, 正解テンソル) のペアを作る関数
loadAdmitData :: FilePath -> Float -> IO (Tensor, Tensor)
loadAdmitData filepath threshold = do
    csvData <- BL.readFile filepath
    case decode HasHeader csvData of
        Left err -> error $ "CSV読み込みエラー: " ++ err
        Right records -> do
            let dataList = V.toList records
                -- 【修正】各特徴量を最大値（概算）で割って正規化 (0~1の範囲に収める)
                inputs = map (\d -> [ gre d / 340.0
                                    , toefl d / 120.0
                                    , rating d / 5.0
                                    , sop d / 5.0
                                    , lor d / 5.0
                                    , cgpa d / 10.0
                                    , research d ]) dataList
                
                targets = map (\d -> if chance d >= threshold then [1.0 :: Float] else [0.0 :: Float]) dataList                
            return ( toDType Float $ asTensor inputs
                   , toDType Float $ asTensor targets )

--------------------------------------------------------------------------------
-- MLP
--------------------------------------------------------------------------------

-- ネットワークの設計図
data MLPSpec = MLPSpec
  { feature_counts :: [Int],   -- 各層のニューロンの数
    nonlinearitySpec :: Tensor -> Tensor     -- 活性化関数
}

-- 実際にメモリ上に存在するネットワークの構造
data MLP = MLP
  { layers :: [Linear],
    nonlinearity :: Tensor -> Tensor
  }
  deriving (Generic, Parameterized)      -- このネットワークの中に入っている重みとバイアスを自動的に見つけてくれる
-- Linear型・・・線形層
-- w * x + b という計算を行い、重みとバイアスを持つ

-- 設計と実体を分ける
-- Haskellが副作用を嫌うから
-- 設計は純粋な値で表現し、実体はランダムに初期化される

-- MLPSpecからMLPを作るためのルール
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

mlp :: MLP -> Tensor -> Tensor
mlp MLP {..} input = foldl' revApply input $ intersperse nonlinearity $ map linear layers
  where
    revApply x f = f x

numIters = 2000

model :: MLP -> Tensor -> Tensor
model params t = mlp params t
main :: IO ()
main = do
  (x, y) <- loadAdmitData "Session5/data/Admission_Predict.csv" 0.7
  
  init <- sample $ MLPSpec
        { feature_counts = [7, 16, 1],
          nonlinearitySpec = Torch.sigmoid
        }
        
  let optimizer = GD -- オプティマイザの初期化

  -- 【修正】オプティマイザの更新状態も引き継ぐようにループの型を変更
  ((trained, _), losses) <- foldLoop ((init, optimizer), []) numIters $ \((state, opt), lossHistory) i -> do
    
    -- 【修正】モデルの出力結果全体に対して最後にsigmoidをかける (intersperseの仕様回避)
    let y' = squeezeAll $ Torch.sigmoid $ model state x
        target = squeezeAll y
        loss = mseLoss target y'
        
    when (i `mod` 100 == 0) $ do
      putStrLn $ "Iteration: " ++ show i ++ " | Loss: " ++ show loss
      
    -- 【修正】新しいオプティマイザ(newOpt)も受け取って次に渡す
    (newState, newOpt) <- runStep state opt loss 1e-1
    
    let currentLoss = asValue loss :: Float 
    return ((newState, newOpt), lossHistory ++ [currentLoss])
  
  putStrLn "モデルの予測結果（最初の5件分）:"
  -- 【修正】予測時にも忘れずに sigmoid をかける
  let finalPreds = squeezeAll $ Torch.sigmoid $ model trained x
  print (sliceDim 0 0 5 1 finalPreds)
  
  putStrLn "実際の正解データ（最初の5件分）:"
  print (sliceDim 0 0 5 1 (squeezeAll y))

  putStrLn "\nTraining Complete!"
  
  putStrLn "学習曲線を生成中..."
  drawLearningCurve "Session5/learning_curve.png" "Admit Model Learning Curve" [("Training Loss", losses)]
  putStrLn "learning_curve.png を保存しました！"

  ```
  <img src="learning_curve.png" width="400">


## 3

```
{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE FunctionalDependencies #-}
{-# LANGUAGE RecordWildCards #-}

module Admit where

import Evaluation

import Control.Monad (when, foldM, replicateM) -- foldMを追加
import Data.List (foldl', intersperse, scanl', transpose)
import GHC.Generics
import Torch hiding (transpose)
import Torch.NN      
import Torch.Optim   
import qualified Data.ByteString.Lazy as BL
import Data.Csv
import qualified Data.Vector as V
import Chart (drawLearningCurve)
--------------------------------------------------------------------------------
-- Data Loading (Kaggle Admission Data)
--------------------------------------------------------------------------------

-- CSVの1行分を表すデータ型
data AdmitData = AdmitData
  { serialNo :: Float
  , gre      :: Float
  , toefl    :: Float
  , rating   :: Float
  , sop      :: Float
  , lor      :: Float
  , cgpa     :: Float
  , research :: Float
  , chance   :: Float
  } deriving (Generic, FromRecord, Show)
  -- FromRecord をつけると、cassavaが自動的にCSVの列と紐づけてくれます


-- CSVファイルを読み込んで、(入力テンソル, 正解テンソル) のペアを作る関数
loadAdmitData :: FilePath -> Float -> IO (Tensor, Tensor)
loadAdmitData filepath threshold = do
    csvData <- BL.readFile filepath
    case decode HasHeader csvData of
        Left err -> error $ "CSV読み込みエラー: " ++ err
        Right records -> do
            let dataList = V.toList records
                -- 【修正】各特徴量を最大値（概算）で割って正規化 (0~1の範囲に収める)
                inputs = map (\d -> [ gre d / 340.0
                                    , toefl d / 120.0
                                    , rating d / 5.0
                                    , sop d / 5.0
                                    , lor d / 5.0
                                    , cgpa d / 10.0
                                    , research d ]) dataList
                
                targets = map (\d -> if chance d >= threshold then [1.0 :: Float] else [0.0 :: Float]) dataList                
            return ( toDType Float $ asTensor inputs
                   , toDType Float $ asTensor targets )

--------------------------------------------------------------------------------
-- MLP
--------------------------------------------------------------------------------

-- ネットワークの設計図
data MLPSpec = MLPSpec
  { feature_counts :: [Int],   -- 各層のニューロンの数
    nonlinearitySpec :: Tensor -> Tensor     -- 活性化関数
}

-- 実際にメモリ上に存在するネットワークの構造
data MLP = MLP
  { layers :: [Linear],
    nonlinearity :: Tensor -> Tensor
  }
  deriving (Generic, Parameterized)      -- このネットワークの中に入っている重みとバイアスを自動的に見つけてくれる
-- Linear型・・・線形層
-- w * x + b という計算を行い、重みとバイアスを持つ

-- 設計と実体を分ける
-- Haskellが副作用を嫌うから
-- 設計は純粋な値で表現し、実体はランダムに初期化される

-- MLPSpecからMLPを作るためのルール
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

mlp :: MLP -> Tensor -> Tensor
mlp MLP {..} input = foldl' revApply input $ intersperse nonlinearity $ map linear layers
  where
    revApply x f = f x

numIters = 2000

model :: MLP -> Tensor -> Tensor
model params t = mlp params t


-- 平均の計算
average :: [Double] -> Double
average xs = sum xs / fromIntegral (length xs)

-- 分散の計算: 各データと平均の差の2乗の平均
variance :: [Double] -> Double
variance xs = 
    let avg = average xs
        squaredDiffs = map (\x -> (x - avg) ^ 2) xs
    in sum squaredDiffs / fromIntegral (length xs)

-- 引数に「実行番号(runId)」を追加
runExperiment :: Int -> IO [Double]
runExperiment runId = do
  (x, y) <- loadAdmitData "Session5/data/Admission_Predict.csv" 0.7
  
  -- MLPの初期化と学習
  init <- sample $ MLPSpec { feature_counts = [7, 16, 1], nonlinearitySpec = Torch.sigmoid }
  
  -- ★ 変更点: losses を記録して受け取るように修正
  ((trained, _), losses) <- foldLoop ((init, GD), []) 2000 $ \((state, opt), lossHistory) i -> do
      let y' = squeezeAll $ Torch.sigmoid $ model state x
          target = squeezeAll y
          loss = mseLoss target y'
      (newState, newOpt) <- runStep state opt loss 1e-1
      let currentLoss = asValue loss :: Float  -- LossをFloatに変換
      return ((newState, newOpt), lossHistory ++ [currentLoss]) -- 履歴に追加
  
  -- ★ 追加: ここで各回の学習曲線を保存！ファイル名に runId を入れる
  let filename = "Session5/learning_curve_run" ++ show runId ++ ".png"
  drawLearningCurve filename ("Learning Curve (Run " ++ show runId ++ ")") [("Training Loss", losses)]
  putStrLn $ "-> " ++ filename ++ " を保存しました"

  -- 予測結果と正解データの Int リスト化
  let probsList = asValue (squeezeAll $ Torch.sigmoid $ model trained x) :: [Float]
      predInts = map (\p -> if p >= 0.7 then 1 else 0) probsList
  let actualInts = map round (asValue (squeezeAll y) :: [Float])
  
  -- 全指標を計算
  let acc   = Evaluation.accuracy actualInts predInts
      prec  = Evaluation.precision 1 actualInts predInts
      rec   = Evaluation.recall 1 actualInts predInts
      f1    = Evaluation.f1Score 1 actualInts predInts
      macF1 = Evaluation.macroF1 2 actualInts predInts
      weiF1 = Evaluation.weightedF1 2 actualInts predInts
      micF1 = Evaluation.microF1 2 actualInts predInts
  
  return [acc, prec, rec, f1, macF1, weiF1, micF1]
main :: IO ()
main = do
  (x, y) <- loadAdmitData "Session5/data/Admission_Predict.csv" 0.7
  
  init <- sample $ MLPSpec
        { feature_counts = [7, 16, 1],
          nonlinearitySpec = Torch.sigmoid
        }
        
  let optimizer = GD -- オプティマイザの初期化

  -- オプティマイザの更新状態も引き継ぐようにループの型を変更
  ((trained, _), losses) <- foldLoop ((init, optimizer), []) numIters $ \((state, opt), lossHistory) i -> do
    
    -- モデルの出力結果全体に対して最後にsigmoidをかける (intersperseの仕様回避)
    let y' = squeezeAll $ Torch.sigmoid $ model state x
        target = squeezeAll y
        loss = mseLoss target y'
        
    -- when (i `mod` 100 == 0) $ do
    --   putStrLn $ "Iteration: " ++ show i ++ " | Loss: " ++ show loss
      
    -- 新しいオプティマイザ(newOpt)も受け取って次に渡す
    (newState, newOpt) <- runStep state opt loss 1e-1
    
    let currentLoss = asValue loss :: Float 
    return ((newState, newOpt), lossHistory ++ [currentLoss])
  
--   putStrLn "モデルの予測結果（最初の5件分）:"
--   -- 予測時にも忘れずに sigmoid をかける
--   let finalPreds = squeezeAll $ Torch.sigmoid $ model trained x
--   print (sliceDim 0 0 5 1 finalPreds)
  
--   putStrLn "実際の正解データ（最初の5件分）:"
--   print (sliceDim 0 0 5 1 (squeezeAll y))

  putStrLn "\nTraining Complete!"

  putStrLn "\n=== モデルの評価（1回目の詳細結果） ==="

  -- 1. 全データをモデルに入れて予測確率を出す
  let probsTensor = squeezeAll $ Torch.sigmoid $ model trained x
  let probsList = asValue probsTensor :: [Float] -- TensorからHaskellのリストへ

  -- 2. 確率を 0.7 を境に 1 と 0 の Int に変換する
  let predInts = map (\p -> if p >= 0.7 then 1 else 0) probsList

  -- 3. 正解データも Int のリストにする
  let actualInts = map (\p -> round p) (asValue (squeezeAll y) :: [Float])

  -- 4. 自作した Evaluation モジュールで全指標を出力！
  putStrLn "[ 混同行列 (Confusion Matrix) ]"
  mapM_ print $ Evaluation.confusionMatrix 2 actualInts predInts

  putStrLn "\n[ 各評価指標 ]"
  putStrLn $ "Accuracy    : " ++ show (Evaluation.accuracy actualInts predInts)
  putStrLn $ "Precision(1): " ++ show (Evaluation.precision 1 actualInts predInts)
  putStrLn $ "Recall(1)   : " ++ show (Evaluation.recall 1 actualInts predInts)
  putStrLn $ "F1 Score(1) : " ++ show (Evaluation.f1Score 1 actualInts predInts)
  putStrLn $ "Macro-F1    : " ++ show (Evaluation.macroF1 2 actualInts predInts)
  putStrLn $ "Weighted-F1 : " ++ show (Evaluation.weightedF1 2 actualInts predInts)
  putStrLn $ "Micro-F1    : " ++ show (Evaluation.microF1 2 actualInts predInts)

  ------------------------------------------------------------------
  -- (この下に「学習曲線を生成中...」や複数回実行のコードが続きます)
  putStrLn "学習曲線を生成中..."
  drawLearningCurve "Session5/learning_curve2.png" "Admit Model Learning Curve" [("Training Loss", losses)]
  putStrLn "learning_curve2.png を保存しました！"  -- ← 今のmainの最後の行

  ------------------------------------------------------------------
  -- 複数回実行による全指標の安定性評価
  ------------------------------------------------------------------
  putStrLn "\n=== 複数回実行による全指標の評価 ==="
  let numRuns = 5  -- 実行回数
  putStrLn $ show numRuns ++ " 回の実験を開始します（少し時間がかかります）..."
  
  -- results は [[Double]] になる
  -- (例: [ [1回目Acc, 1回目Prec...], [2回目Acc, 2回目Prec...] ])
  results <- mapM runExperiment [1..numRuns]
  
  -- transpose を使うと、指標ごとにリストをまとめ直せる！
  -- transposed = [ [全回のAcc], [全回のPrec], [全回のRec]... ]
  let transposed = transpose results
  
  -- 各指標ごとに平均と分散を計算
  let avgAcc   = average (transposed !! 0)
      varAcc   = variance (transposed !! 0)
      avgPrec  = average (transposed !! 1)
      varPrec  = variance (transposed !! 1)
      avgRec   = average (transposed !! 2)
      varRec   = variance (transposed !! 2)
      avgF1    = average (transposed !! 3)
      varF1    = variance (transposed !! 3)
      avgMacF1 = average (transposed !! 4)
      varMacF1 = variance (transposed !! 4)
      avgWeiF1 = average (transposed !! 5)
      varWeiF1 = variance (transposed !! 5)
      avgMicF1 = average (transposed !! 6)
      varMicF1 = variance (transposed !! 6)

  putStrLn "\n[ 平均 (Average)  /  分散 (Variance) ]"
  putStrLn $ "Accuracy    : " ++ show avgAcc ++ "  /  " ++ show varAcc
  putStrLn $ "Precision(1): " ++ show avgPrec ++ "  /  " ++ show varPrec
  putStrLn $ "Recall(1)   : " ++ show avgRec ++ "  /  " ++ show varRec
  putStrLn $ "F1 Score(1) : " ++ show avgF1 ++ "  /  " ++ show varF1
  putStrLn $ "Macro-F1    : " ++ show avgMacF1 ++ "  /  " ++ show varMacF1
  putStrLn $ "Weighted-F1 : " ++ show avgWeiF1 ++ "  /  " ++ show varWeiF1
  putStrLn $ "Micro-F1    : " ++ show avgMicF1 ++ "  /  " ++ show varMicF1

  ```

  ```
  === モデルの評価（1回目の詳細結果） ===
[ 混同行列 (Confusion Matrix) ]
[125,28]
[73,174]

[ 各評価指標 ]
Accuracy    : 0.7475
Precision(1): 0.8613861386138614
Recall(1)   : 0.7044534412955465
F1 Score(1) : 0.7750556792873051
Macro-F1    : 0.7436531957690087
Weighted-F1 : 0.7510327793958084
Micro-F1    : 0.7475

```

```
[ 平均 (Average)  /  分散 (Variance) ]
Accuracy    : 0.7505  /  8.499999999999883e-6
Precision(1): 0.8608256406673525  /  9.117346943501542e-6
Recall(1)   : 0.7109311740890687  /  6.949794292645338e-5
F1 Score(1) : 0.7786881134292571  /  1.6044121900775502e-5
Macro-F1    : 0.7463663951036084  /  6.392536013837645e-6
Weighted-F1 : 0.7539619989101359  /  8.161442255418451e-6
Micro-F1    : 0.7505  /  8.499999999999883e-6

```

<img src="learning_curve_run1.png" width="400">
<img src="learning_curve_run2.png" width="400">
<img src="learning_curve_run3.png" width="400">
<img src="learning_curve_run4.png" width="400">
<img src="learning_curve_run5.png" width="400">