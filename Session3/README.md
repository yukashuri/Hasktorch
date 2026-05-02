# Hands-on Tasks

## 3-b Result of the Linear Function
```
correct answer: 130
estimated: 176.72504
******
correct answer: 195
estimated: 197.81503
******
correct answer: 218
estimated: 249.43002
******
correct answer: 166
estimated: 193.93002
******
correct answer: 163
estimated: 214.46503
******
correct answer: 155
estimated: 165.07004
******
correct answer: 204
estimated: 178.94504
******
correct answer: 270
estimated: 203.36502
******
correct answer: 205
estimated: 164.51503
******
correct answer: 127
estimated: 137.87503
******
correct answer: 260
estimated: 211.69003
******
correct answer: 249
estimated: 238.33002
******
correct answer: 251
estimated: 236.11005
******
correct answer: 158
estimated: 158.41003
******
correct answer: 167
estimated: 190.60004
******
```

```
import Torch.Tensor (Tensor, asTensor)
import Torch.Functional (mean, add, mul)

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

main :: IO ()
main = do
  -- Below are pseudo code
  let sampleA = asTensor (0.555 :: Float)
  let sampleB = asTensor (94.585026 :: Float)
	
  -- Iterate through the provided xs and ys data. 
  -- For each pair, convert x to a tensor, calculate the estimatedY using your linear function with the provided sampleA and sampleB, and print both the correct y and the estimatedY.
  
  let estimatedYs = linear (sampleA, sampleB) xs

  putStrLn $ "correct answer: "
  print ys

  putStrLn $ "estimated: "
  print estimatedYs
    
    -- Expected outputs:
    -- correct answer: 148
    -- estimated: ?
    -- *******
    -- correct answer: 186
    -- ...
  return ()
```

## 3-d Define a cost function
```
-- 予測値がどれくらいずれているか計算する関数
cost ::
    Tensor -> -- ^ grand truth: 1 × 10
    Tensor -> -- ^ estimated values: 1 × 10
    Tensor    -- ^ loss: scalar
cost z z' = 
    let diff = z - z' 
        squared = mul diff diff                 -- 要素ごとの掛け算
    in  mean squared
```

## 3-e 
