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