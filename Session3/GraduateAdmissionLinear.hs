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
  
-----------------------------------------------------
-- ここから評価

  putStrLn "\n--- Evaluation Phase (eval.csv) ---"
  
  -- 1. eval.csv の読み込み
  evalData <- loadCSV "Session3/data/eval.csv"
  
  -- 2. 学習データと同じように x を 340.0 で正規化してリスト化
  let rawEvalXs = map (\d -> realToFrac (greScore d) / 340.0 :: Float) (V.toList evalData)
      rawEvalYs = map (\d -> realToFrac (chanceOfAdmit d) :: Float) (V.toList evalData)

  -- 3. Tensor に変換
  let evalXs = asTensor rawEvalXs
      evalYs = asTensor rawEvalYs

  -- 4. 学習で得た最終パラメータ (finalA, finalB) を使って予測値を計算
  let evalPredictions = linear (finalA, finalB) evalXs
  
  -- 5. 評価データにおける Loss (MSE) を計算
  -- 誤差の平均
  let evalLoss = cost evalYs evalPredictions
  putStrLn $ "Evaluation Loss (MSE): " ++ show (asValue evalLoss :: Float)

  -- 6. 実際の値と予測値を比較（最初の5件だけピックアップして表示）
  putStrLn "\n--- Sample Predictions (Actual vs Estimated) ---"
  let actualYsList = asValue evalYs :: [Float]
  let estimatedYsList = asValue evalPredictions :: [Float]
  
  -- zipで正解と予測のペアを作り、forM_でループして表示
  forM_ (take 5 (zip actualYsList estimatedYsList)) $ \(actual, estimated) -> do
    putStrLn $ "Actual (正解):    " ++ show actual
    putStrLn $ "Estimated (予測): " ++ show estimated
    putStrLn "******"

  putStrLn "\nプログラムが正常に完了しました！"

  -- ==========================================
  -- 検証（Validation）フェーズ：valid.csv の確認
  -- ==========================================
  putStrLn "\n--- Validation Phase (valid.csv) ---"
  
  -- 1. valid.csv の読み込み
  validData <- loadCSV "Session3/data/valid.csv"
  
  -- 2. x を 340.0 で正規化してリスト化
  let rawValidXs = map (\d -> realToFrac (greScore d) / 340.0 :: Float) (V.toList validData)
      rawValidYs = map (\d -> realToFrac (chanceOfAdmit d) :: Float) (V.toList validData)

  -- 3. Tensor に変換
  let validXs = asTensor rawValidXs
      validYs = asTensor rawValidYs

  -- 4. 学習済みパラメータで予測
  let validPredictions = linear (finalA, finalB) validXs
  
  -- 5. valid.csv での Loss を計算
  let validLoss = cost validYs validPredictions
  putStrLn $ "Validation Loss (MSE): " ++ show (asValue validLoss :: Float)

  -- 6. 結果のサンプル表示
  putStrLn "\n--- Sample Predictions (Valid Data) ---"
  let actualValidYsList = asValue validYs :: [Float]
  let estimatedValidYsList = asValue validPredictions :: [Float]
  
  forM_ (take 5 (zip actualValidYsList estimatedValidYsList)) $ \(actual, estimated) -> do
    putStrLn $ "Actual (正解):    " ++ show actual
    putStrLn $ "Estimated (予測): " ++ show estimated
    putStrLn "******"