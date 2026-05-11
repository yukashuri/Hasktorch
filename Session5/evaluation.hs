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

