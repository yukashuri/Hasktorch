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



-- -- パラメータを更新する関数
-- updateStep :: Float -> (Tensor, Tensor) -> Int -> (Tensor, Tensor)
-- updateStep learningRate (a, b) n = 

-- -- 指定した回数だけupdateStepを繰り返し適用する関数
-- train :: Int -> Float -> (Tensor, Tensor)


main :: IO ()
main = do
  -- Below are pseudo code
  let sampleA = asTensor (0.555 :: Float)
  let sampleB = asTensor (94.585026 :: Float)
	
  -- Iterate through the provided xs and ys data. 
  -- For each pair, convert x to a tensor, calculate the estimatedY using your linear function with the provided sampleA and sampleB, and print both the correct y and the estimatedY.
  
  let estimatedYs = linear (sampleA, sampleB) xs

  let ysList = asValue ys :: [Float]
  let estimatedYsList = asValue estimatedYs :: [Float]
  
  forM_ (zip ysList estimatedYsList) $ \(correct, estimated) -> do
    -- 出力例の「148」のように整数表示にするため、正解側だけ round (四捨五入) で整数化しています
    putStrLn $ "correct answer: " ++ show (round correct :: Int)
    putStrLn $ "estimated: " ++ show estimated
    putStrLn "******"
    
    -- Expected outputs:
    -- correct answer: 148
    -- estimated: ?
    -- *******
    -- correct answer: 186
    -- ...
  return ()