module Main (main) where


--hasktorch
import Torch
import System.Random
import Data.Int
import Data.List (nub)

-- 21. Create a 5x5 matrix with row values ranging from 0 to 4 (★★☆)
-- ans = [[0,1,2,3,4],
--        [0,1,2,3,4],
--        [0,1,2,3,4],
--        [0,1,2,3,4],
--        [0,1,2,3,4]]
q21_ans :: Tensor
q21_ans = reshape [5,5] (asTensor ([j | i <- [0 .. 4], j <- [0 .. 4]] :: [Int]))

-- 22. Consider a generator function that generates 10 integers and use it to build an array (★☆☆)
-- 10個の整数を生成する関数を考え、それを使って配列を作成する
q22_make_int = Prelude.take 10 (randomRs (-100, 100) (mkStdGen 6) :: [Int])
q22_ans = asTensor q22_make_int

-- 23. Create a vector of size 10 with values ranging from 0 to 1, both excluded (★★☆)
-- 0から１で等間隔に並んだ１０個の少数のリスト。両端を含まない。→０から１を11等分する
q23_ans = [ i/11 | i <- [1 .. 10]]

-- 24. Create a random vector of size 10 and sort it (★★☆)
q24 = q22_make_int
quicksort :: (Ord a) => [a] -> [a]
quicksort [] = []
quicksort (x:xs) = 
    let smallerOrEqual = [a| a <- xs, a <= x]
        larger = [a | a <- xs, a > x]
        in quicksort smallerOrEqual ++ [x] ++ quicksort larger
q24_ans = quicksort q24

-- 25. How to sum a small array faster than Torch.Tensor.sum? (★★☆)
-- テンソル用のTorch.sumではなくて、Haskell標準のsumを使ほうが、小さな配列の計算は早く処理できる。

-- 26. Consider two random array A and B, check if they are equal (★★☆)
q26_a = Prelude.take 10 (randomRs (-100, 100) (mkStdGen 9) :: [Int])
q26_b = Prelude.take 10 (randomRs (-100, 100) (mkStdGen 10) :: [Int])
q26_ans_1 = q26_a == q26_b
q26_ans_2 = q26_a == q26_a

-- 27. Make an array immutable (read-only) (★★☆)
-- Haskellは何もしなくてもimmutable

-- 28. Consider a random 10x2 matrix representing cartesian coordinates, convert them to polar coordinates (★★☆)
q28 = reshape [10, 2] (asTensor (Prelude.take 20 (randoms (mkStdGen 28)) :: [Float]))
-- XとYの行列を、距離(r)と角度(θ)の行列に変換するコード
q28_ans = asTensor [ [Prelude.sqrt (x**2 + y**2), atan2 y x] | [x, y] <- asValue q28 :: [[Float]] ]

-- 29. Create random vector of size 10 and replace the maximum value by 0 (★★☆)
-- 10次元のランダムなベクトルの最大値を0に置き換える
q29 = Prelude.take 10 (randomRs (-100, 100) (mkStdGen 0) :: [Int])
q29_ans = [if x == Prelude.maximum q29 then 0 else x | x <- q29]

-- 30. Create a structured array with x and y coordinates covering the [0,1]x[0,1] area (★★☆)
q30_x = [i/10 | i <- [0..10]]
q30_y = [j/10 | j <- [0..10]]
q30_ans = [[x,y] | x <- q30_x, y <- q30_y]

-- 31. Given two arrays, X and Y, construct the Cauchy matrix C (Cij =1/(xi - yj))
q31_x = [i/2 | i <- [0..2]]
q31_y = [j/2 | j <- [0..2]]
q31_ans = [[1/(x-y) | y <- q31_y] | x <- q31_x]

-- 32. Print the minimum and maximum representable value for each numpy scalar type (★★☆)
-- 限界値を教えてくれる関数 最小値：minBound 最大値：maxBound
q32_8_min = (minBound :: Int8)
q32_8_max = (maxBound :: Int8)
q32_16_min = (minBound :: Int16)
q32_16_max = (maxBound :: Int16)
q32_32_min = (minBound :: Int32)
q32_32_max = (maxBound :: Int32)
q32_64_min = (minBound :: Int64)
q32_64_max = (maxBound :: Int64)
q32_ans = (q32_8_min, q32_8_max, q32_16_min, q32_16_max, q32_32_min, q32_32_max, q32_64_min, q32_64_max)

-- 33. How to print all the values of an array? (★★☆)
q33_huge_data = asTensor [1..1000 :: Float]

-- 34. How to find the closest value (to a given scalar) in a vector? (★★☆)
-- ベクトルの中から、指定したスカラ値に一番近い値を見つけるには？
q34_vector = [10.0, 25.0, 40.0, 55.0, 70.0] :: [Float]
q34_scalar = 38.0 :: Float
-- 差が一番小さくなるものを持ってくる
q34_ans = snd (minimum [(Prelude.abs (x - q34_scalar), x) | x <- q34_vector])

-- 35. Create a structured array representing a position (x,y) and a color (r,g,b) (★★☆)
-- 位置(x,y)と、色(r,g,b)のデータを持つ構造化された配列を作成せよ。
-- [ ((x,y), (r,g,b)), ((x,y), (r,g,b)), ...]
data Point = Point {
    x :: Float, 
    y :: Float,
    r :: Int, 
    g :: Int,
    b :: Int
} deriving (Show)
q35_ans = [ Point { x = 0.0, y = 1.5, r = 255, g = 0, b = 0}]

-- 36. Consider a random vector with shape (100,2) representing coordinates, find point by point distances (★★☆)
-- 座標を表す(100,2)のランダムなベクトルについて、点と点の全ての距離を求めよ
q36 = reshape [100, 2] (asTensor (Prelude.take 200 (randoms (mkStdGen 28)) :: [Float]))
q36_ans = [[ Prelude.sqrt ((x1 - x2) ** 2 + (y1 - y2) ** 2) | [x1, y1] <- asValue q36 :: [[Float]]] | [x2, y2] <- asValue q36 :: [[Float]]]

-- 37. How to convert a float (32 bits) array into an integer (32 bits) in place?
-- 32ビットの小数の配列を32ビットの整数に変換するには？
-- Haskellは上書き禁止なので、FloatからIntへ型変換した新しい配列を作る


-- 38. What is the equivalent of enumerate for numpy arrays? (★★☆)
-- NumPy配列において、Pythonのenumerateに相当する機能は何か？
enum :: [a] -> [(Int, a)]
enum lst = xs
    where xs = zip [0..] lst 
q38_ans = enum [1,4,5,2]

-- 39. Generate a generic 2D Gaussian-like array (★★☆)
-- 一般的な二次元のガウス分布のような配列を生成せよ。
q39_xs = [-2.0, -1.5 .. 2.0] :: [Float]
q39_ys = [-2.0, -1.5 .. 2.0] :: [Float]
q39_ans = [[Prelude.exp (-(x**2 + y**2)) | x <- q39_xs] | y <- q39_ys]

-- 40. How to randomly place p elements in a 2D array? (★★☆)
-- 二次元配列の中に、p個の要素をランダムに配置するには。
q40_size = 5 :: Int
q40_p = 3 :: Int
q40_coords = Prelude.take q40_p (nub (randomRs ((1, 1), (q40_size, q40_size)) (mkStdGen 42))) :: [(Int, Int)]
q40_ans = [ [ if (x, y) `elem` q40_coords then 1 else 0 | x <- [1..q40_size] ] | y <- [1..q40_size] ]



main :: IO ()
main = do
    
    print q21_ans
    print q22_ans
    print q23_ans
    print q24
    print q24_ans
    print q26_ans_1
    print q26_ans_2
    print q28_ans
    print q29
    print q29_ans
    print q30_ans
    print q31_ans
    print q32_ans
    -- print q33_huge_data
    -- print (asValue q33_huge_data :: [Float])
    print q34_ans
    print q35_ans
    -- print q36_ans
    print q38_ans
    print q39_ans
    print q40_ans