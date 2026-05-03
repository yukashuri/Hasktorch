{-# LANGUAGE OverloadedStrings #-}

module GraduateAdmissionLinear where

import Data.Csv
import qualified Data.ByteString.Lazy as BL
import qualified Data.Vector as V
import Torch.Tensor (Tensor, asTensor, asValue)
import Torch.Functional (mean, add, mul)
import Control.Monad (forM_)

-- 1. データ型の定義（先ほどのまま）
data AdmissionData = AdmissionData
  { greScore      :: Double
  , chanceOfAdmit :: Double
  } deriving (Show)

instance FromNamedRecord AdmissionData where
  parseNamedRecord r = AdmissionData
    <$> r .: "GRE Score"
    <*> r .: "Chance of Admit "

loadCSV :: FilePath -> IO (V.Vector AdmissionData)
loadCSV path = do
  csvData <- BL.readFile path
  case decodeByName csvData of
    Left err -> do
      putStrLn $ path ++ " の読み込みエラー: " ++ err
      return V.empty
    Right (_, v) -> return v

-- 2. 予測関数とコスト関数
linear :: (Tensor, Tensor) -> Tensor -> Tensor
linear (slope, intercept) input = add (mul slope input) intercept

cost :: Tensor -> Tensor -> Tensor
cost z z' = 
    let diff = z - z' 
        squared = mul diff diff
    in  mean squared

-- 3. 勾配の計算とパラメータ更新
-- ★ xs と ys を外部から受け取るように引数を追加しています
calculateNewA :: Tensor -> Tensor -> Tensor -> Tensor -> Tensor
calculateNewA a estimatedYs xs ys = a - rate * gradA
  where
    diff = estimatedYs - ys
    rate = asTensor (0.1 :: Float) -- 正規化に伴い、学習率を少し大きめに調整
    gradA = mean (mul (asTensor (2.0 :: Float)) (mul diff xs))

calculateNewB :: Tensor -> Tensor -> Tensor -> Tensor
calculateNewB b estimatedYs ys = b - rate * gradB
  where
    diff = estimatedYs - ys
    rate = asTensor (0.1 :: Float)
    gradB = mean (mul (asTensor (2.0 :: Float)) diff)

-- 4. 学習ループ
-- ★ xs と ys を受け取り、次のループへ引き継ぎます
trainLoop :: Int -> Int -> Tensor -> Tensor -> Tensor -> Tensor -> [Float] -> IO (Tensor, Tensor, [Float])
trainLoop currentEpoch maxEpoch a b xs ys history
  | currentEpoch > maxEpoch = do
      putStrLn "Training completed!"
      return (a, b, reverse history)
  | otherwise = do
      let estimatedYs = linear (a, b) xs
          currentLoss = cost ys estimatedYs  
          lossValue   = asValue currentLoss :: Float
      
      -- 10エポックごとに途中経過を表示
      if currentEpoch `mod` 10 == 0
         then putStrLn $ "Epoch " ++ show currentEpoch ++ " - Loss: " ++ show lossValue
         else return ()
      
      let newA = calculateNewA a estimatedYs xs ys
          newB = calculateNewB b estimatedYs ys
      
      trainLoop (currentEpoch + 1) maxEpoch newA newB xs ys (lossValue : history)

-- 5. メイン処理（プログラムのスタート地点）
main :: IO ()
main = do
  putStrLn "--- Graduate Admission Linear Regression ---"
  putStrLn "データを読み込んでいます..."
  
  trainData <- loadCSV "Session3/data/train.csv"
  
  -- Vector内のデータを抽出してリスト化
  -- ★ ここで GRE Score を 340.0 で割って 0〜1.0 の範囲に「正規化」しています
  let rawXs = map (\d -> realToFrac (greScore d) / 340.0 :: Float) (V.toList trainData)
      rawYs = map (\d -> realToFrac (chanceOfAdmit d) :: Float) (V.toList trainData)

  -- Tensorに変換
  let xs = asTensor rawXs
      ys = asTensor rawYs

  -- パラメータの初期値
  let sampleA = asTensor (0.0 :: Float)
      sampleB = asTensor (0.0 :: Float)
    
  putStrLn "--- Training Started ---"
  
  -- 例として100エポック回します（必要に応じて変更してください）
  (finalA, finalB, losses) <- trainLoop 1 100 sampleA sampleB xs ys []
  
  putStrLn "------------------------"
  putStrLn $ "Final a (slope): " ++ show (asValue finalA :: Float)
  putStrLn $ "Final b (intercept): " ++ show (asValue finalB :: Float)
  
  -- ★ 課題にある「eval.csv を使った予測と検証」は、この下に書き足していきます