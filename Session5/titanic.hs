{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE MultiParamTypeClasses #-}

module Titanic where

import GHC.Generics
import Torch
import qualified Data.ByteString.Lazy as BL
import Data.Csv
import qualified Data.Vector as V

import Control.Monad (when, foldM)
import Data.List (foldl', intersperse, scanl')
import Torch.NN      
import Torch.Optim

import Evaluation

import Chart (drawLearningCurve, drawConfusionMatrix)

-- 1. タイタニックのCSVの1行分を表すデータ型
-- 欠損値がある列（Age, Cabin, Embarked）は `Maybe` で囲むのがポイント！
data TitanicRow = TitanicRow
  { passengerId :: Int
  , survived    :: Float  -- 正解ラベル (0.0 or 1.0)
  , pclass      :: Float
  , name        :: String
  , sex         :: String
  , age         :: Maybe Float  -- 欠損値あり
  , sibSp       :: Float
  , parch       :: Float
  , ticket      :: String
  , fare        :: Float
  , cabin       :: Maybe String -- 欠損値多数
  , embarked    :: Maybe String -- 欠損値あり
  } deriving (Generic, Show)

-- CSVのヘッダー名とフィールドを紐づける
instance FromNamedRecord TitanicRow where
  parseNamedRecord r = TitanicRow
    <$> r .: "PassengerId"
    <*> r .: "Survived"
    <*> r .: "Pclass"
    <*> r .: "Name"
    <*> r .: "Sex"
    <*> r .: "Age"
    <*> r .: "SibSp"
    <*> r .: "Parch"
    <*> r .: "Ticket"
    <*> r .: "Fare"
    <*> r .: "Cabin"
    <*> r .: "Embarked"


-- 2. データの前処理関数 (Preprocessing)
preprocess :: [TitanicRow] -> [( [Float], [Float] )]
preprocess rows = do
    row <- rows
    
    -- ① 欠損値の補完 (Fill missing data)
    -- Ageの欠損値は、タイタニックの平均年齢に近い 29.6 で埋める
    let ageVal = case age row of
                    Just a  -> a
                    Nothing -> 29.6
                    
    -- ② 非数値データの数値化 (Categorical to Numerical)
    -- Sex: female -> 1.0, male -> 0.0
    let sexVal = if sex row == "female" then 1.0 else 0.0
    
    -- Embarked: C -> 0.0, Q -> 1.0, S -> 2.0 (欠損値は一番多いSで埋める)
    let embarkedVal = case embarked row of
                        Just "C" -> 0.0
                        Just "Q" -> 1.0
                        _        -> 2.0

    -- ③ 正規化 (Normalization)
    -- 値のスケールを0.0〜1.0付近に揃える
    let pclassNorm   = pclass row / 3.0
    let ageNorm      = ageVal / 80.0
    let sibspNorm    = sibSp row / 8.0
    let parchNorm    = parch row / 6.0
    let fareNorm     = fare row / 500.0
    let embarkedNorm = embarkedVal / 2.0

    -- ④ 不要なカラムの削除 (Feature selection)
    -- PassengerId, Name, Ticket, Cabin は学習に使わないので特徴量リストに入れない
    let features = [ pclassNorm, sexVal, ageNorm, sibspNorm, parchNorm, fareNorm, embarkedNorm ]
    let target   = [ survived row ]
    
    return (features, target)

-- 3. CSVを読み込んでテンソルに変換する関数
loadTitanicData :: FilePath -> IO (Tensor, Tensor)
loadTitanicData filepath = do
    csvData <- BL.readFile filepath
    case decodeByName csvData of
        Left err -> error $ "CSV読み込みエラー: " ++ err
        Right (_, records) -> do
            let dataList = V.toList records
                processedData = preprocess dataList
                inputs  = map fst processedData
                targets = map snd processedData
            return ( toDType Float $ asTensor inputs
                   , toDType Float $ asTensor targets )

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
  deriving (Generic, Parameterized)

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


-- テスト用の main 関数
main :: IO ()
main = do
    putStrLn "データを読み込んで前処理します..."
    (x, y) <- loadTitanicData "Session5/data/train.csv" -- ← CSVのパスは適宜合わせてください
    
    putStrLn $ "入力データの形状 (サンプル数, 特徴量数): " ++ show (shape x)
    putStrLn $ "正解データの形状 (サンプル数, 1): " ++ show (shape y)
    
    putStrLn "\n前処理済みデータの最初の1件:"
    print (sliceDim 0 0 1 1 x)
    print (sliceDim 0 0 1 1 y)

    putStrLn "\n=== モデルの初期化と学習開始 ==="
    
    -- 入力層7 -> 隠れ層16 -> 出力層1 のMLPを作る
    init <- sample $ MLPSpec { feature_counts = [7, 16, 1], nonlinearitySpec = Torch.sigmoid }
    
    let optimizer = GD
    
    -- 学習ループ
    ((trained, _), losses) <- foldLoop ((init, optimizer), []) numIters $ \((state, opt), lossHistory) i -> do
        
        -- 予測とLoss(誤差)の計算
        let y' = squeezeAll $ Torch.sigmoid $ model state x
            target = squeezeAll y
            
            loss = mseLoss target y'
            
        -- 100回ごとに画面に途中経過を出力
        when (i `mod` 100 == 0) $ do
            putStrLn $ "Iteration: " ++ show i ++ " | Loss: " ++ show loss
            
        -- 重みの更新
        (newState, newOpt) <- runStep state opt loss 1e-1
        
        let currentLoss = asValue loss :: Float 
        return ((newState, newOpt), lossHistory ++ [currentLoss])
    
    putStrLn "\nTraining Complete!"

    putStrLn "\n=== モデルの評価（1回勝負！） ==="
    let probsList = asValue (squeezeAll $ Torch.sigmoid $ model trained x) :: [Float]
        -- 生存確率は 0.5 を境目にする
        predInts = map (\p -> if p >= 0.5 then 1 else 0) probsList
        actualInts = map round (asValue (squeezeAll y) :: [Float])

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
    -- ここから描画処理を追加
    ------------------------------------------------------------------
    -- 1. 予測結果(predInts)と正解データ(actualInts)のリストを作成
    let probsList = asValue (squeezeAll $ Torch.sigmoid $ model trained x) :: [Float]
        -- タイタニックの生存判定のしきい値は通常 0.5 を使います
        predInts = map (\p -> if p >= 0.5 then 1 else 0) probsList
        actualInts = map round (asValue (squeezeAll y) :: [Float])

    -- 2. 学習曲線 (Learning Curve) の PNG を保存
    putStrLn "\n学習曲線を生成中..."
    drawLearningCurve "Session5/titanic_learning_curve.png" "Titanic Learning Curve" [("Training Loss", losses)]
    putStrLn "-> titanic_learning_curve.png を保存しました！"

    -- 3. 混同行列 (Confusion Matrix) の PNG を保存
    putStrLn "混同行列の画像を生成中..."
    -- drawConfusionMatrix は (予測値, 正解値) のペアのリストを要求するので、zip を使ってペアを作ります
    let confusionPairs = zip predInts actualInts
    drawConfusionMatrix "Session5/titanic_confusion_matrix.png" 2 confusionPairs
    putStrLn "-> titanic_confusion_matrix.png を保存しました！"