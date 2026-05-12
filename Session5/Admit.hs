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