module LinearRegression where

import Torch.Tensor (Tensor, asTensor, asValue)
import Torch.Functional (mean, add, mul)
import Control.Monad (forM_)

ys :: Tensor
ys = asTensor ([130, 195, 218, 166, 163, 155, 204, 270, 205, 127, 260, 249, 251, 158, 167] :: [Float])
xs :: Tensor
xs = asTensor ([148, 186, 279, 179, 216, 127, 152, 196, 126, 78, 211, 259, 255, 115, 173] :: [Float])

-- 予測値を返す
linear :: 
    (Tensor, Tensor) -> -- ^ parameters ([a, b]: 1 × 2, c: scalar)
    Tensor ->           -- ^ data x: 1 × 10
    Tensor              -- ^ z: 1 × 10
linear (slope, intercept) input = 
    let y = mul slope input
    in add y intercept

-- 予測値がどれくらいずれているか計算する関数
cost ::
    Tensor -> -- ^ grand truth: 1 × 10
    Tensor -> -- ^ estimated values: 1 × 10
    Tensor    -- ^ loss: scalar
cost z z' = 
    let diff = z - z' 
        squared = mul diff diff                 -- 要素ごとの掛け算
    in  mean squared

calculateNewA :: 
     Tensor ->
     Tensor ->
     Tensor
calculateNewA a y = a - rate * gradA
  where
    diff = y - ys
    rate = 0.00001 -- 学習率
    gradA = mean ( mul 2 (mul diff xs) )-- 勾配の計算（コスト関数のaに対する偏微分）

calculateNewB :: 
     Tensor ->
     Tensor ->
     Tensor
calculateNewB b y = b - rate * gradB
  where
    diff = y - ys
    rate = 0.00001 -- 学習率
    gradB = mean ( mul 2 diff ) -- 勾配の計算（コスト関数のbに対する偏微分）

-- fまでのコードは以下を使う
-- -- 指定したエポック数だけ学習を繰り返す再帰関数
-- trainLoop :: Int -> Int -> Tensor -> Tensor -> IO (Tensor, Tensor)
-- trainLoop currentEpoch maxEpoch a b 
--   | currentEpoch > maxEpoch = do
--       putStrLn "Training completed!"
--       return (a, b)  -- 目標エポックに達したら、最終的なパラメータを返す
--   | otherwise = do
--       -- 1. 現在の a と b で予測値と、ズレ（Loss）を計算
--       let estimatedYs = linear (a, b) xs
--           currentLoss = cost ys estimatedYs  
      
--       -- 2. 現在のエポック数と Loss を画面に表示
--       putStrLn $ "Epoch " ++ show currentEpoch ++ " - Loss: " ++ show (asValue currentLoss :: Float)
      
--       -- 3. a と b を更新（※前回修正した asTensor や sub を使った関数を呼び出します）
--       let newA = calculateNewA a estimatedYs
--           newB = calculateNewB b estimatedYs

--       putStrLn $ "Updated parameters - a: " ++ show (asValue newA :: Float) ++ ", b: " ++ show (asValue newB :: Float)
      
--       -- 4. 新しくなったパラメータを渡して、次のエポックへ進む（再帰呼び出し）
--       trainLoop (currentEpoch + 1) maxEpoch newA newB

-- ★ グラフ用のLoss履歴を返すようにだけ修正
trainLoop :: Int -> Int -> Tensor -> Tensor -> [Float] -> IO (Tensor, Tensor, [Float])
trainLoop currentEpoch maxEpoch a b history
  | currentEpoch > maxEpoch = do
      putStrLn "Training completed!"
      return (a, b, reverse history)  -- 履歴をひっくり返して返す
  | otherwise = do
      let estimatedYs = linear (a, b) xs
          currentLoss = cost ys estimatedYs  
          lossValue   = asValue currentLoss :: Float
      
      putStrLn $ "Epoch " ++ show currentEpoch ++ " - Loss: " ++ show lossValue
      
      let newA = calculateNewA a estimatedYs
          newB = calculateNewB b estimatedYs

      putStrLn $ "Updated parameters - a: " ++ show (asValue newA :: Float) ++ ", b: " ++ show (asValue newB :: Float)
      
      -- 次のループへ
      trainLoop (currentEpoch + 1) maxEpoch newA newB (lossValue : history)

-- fまでのコードは以下を使う
-- main :: IO ()
-- main = do
--   -- Below are pseudo code
--   let sampleA = asTensor (0.555 :: Float)
--   let sampleB = asTensor (94.585026 :: Float)
	
--   -- Iterate through the provided xs and ys data. 
--   -- For each pair, convert x to a tensor, calculate the estimatedY using your linear function with the provided sampleA and sampleB, and print both the correct y and the estimatedY.
--   putStrLn "--- Training Started ---"
  
--   -- ★ ここで1エポック目から10エポック目まで学習ループを回す
--   (finalA, finalB) <- trainLoop 1 10 sampleA sampleB
  
--   putStrLn "------------------------"
--   putStrLn "Final Predictions:"

--   let estimatedYs = linear (finalA, finalB) xs

--   let ysList = asValue ys :: [Float]
--   let estimatedYsList = asValue estimatedYs :: [Float]
  
--   forM_ (zip ysList estimatedYsList) $ \(correct, estimated) -> do
--     -- 出力例の「148」のように整数表示にするため、正解側だけ round (四捨五入) で整数化しています
--     putStrLn $ "correct answer: " ++ show (round correct :: Int)
--     putStrLn $ "estimated: " ++ show estimated
--     putStrLn "******"
    
--     -- Expected outputs:
--     -- correct answer: 148
--     -- estimated: ?
--     -- *******
--     -- correct answer: 186
--     -- ...

  
--   return ()