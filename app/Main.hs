module Main where

import Torch.Tensor (asTensor, asValue)
import Control.Monad (forM_)

-- ★ 分割したファイルを読み込む
import LinearRegression 
import Chart (drawLearningCurve)

main :: IO ()
main = do
  let sampleA = asTensor (0.555 :: Float)
  let sampleB = asTensor (94.585026 :: Float)
    
  putStrLn "--- Training Started ---"
  
  -- LinearRegression.hs の trainLoop を呼び出す
  (finalA, finalB, losses) <- trainLoop 1 100 sampleA sampleB []
  
  putStrLn "------------------------"
  putStrLn "Final Predictions:"

  -- LinearRegression.hs の linear や xs, ys を使う
  let estimatedYs = linear (finalA, finalB) xs

  let ysList = asValue ys :: [Float]
  let estimatedYsList = asValue estimatedYs :: [Float]
  
  forM_ (zip ysList estimatedYsList) $ \(correct, estimated) -> do
    putStrLn $ "correct answer: " ++ show (round correct :: Int)
    putStrLn $ "estimated: " ++ show estimated
    putStrLn "******"

  putStrLn "------------------------"
  putStrLn "Drawing Learning Curve..."
  
  -- ★ ML.Exp.Chart を使ってグラフを描画
  let chartData = ("Linear Regression", losses)
  drawLearningCurve "learning_curve-3.png" "Loss over Epochs" [chartData]
  
  putStrLn "learning_curve.png was successfully created!"