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