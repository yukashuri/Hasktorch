module Main (main) where

--hasktorch
import Torch
import System.Random
import Data.List (transpose)

--1. Create a null vector of size 10 but the fifth value is 1 (★☆☆)
q01_vector = [if i == 4 then 1 else 0 | i <- [0..9]]

--2. Create a vector with values ranging from 10 to 49 (★☆☆)
q02_vector = [i | i <- [10..49]]

--3. Reverse a vector (first element becomes last) (★☆☆)
q03_vector = reverse q02_vector

--4. Create a 3x3 matrix with values ranging from 0 to 8 (★☆☆)
q04_matrix :: Tensor
q04_matrix = reshape [3, 3] (asTensor ([0..8] :: [Float]))
--[0,1,2,3,4,5,6,7,8]を3*3の行列に変形

--5. Find indices of non-zero elements from [1,2,0,0,4,0] (★☆☆)
xs = [1, 2, 0, 0, 4, 0]
paired = zip [0..] xs
--[(0,1), (1,2), (2,0), (3,0), (4,4), (5,0)]
filtered = filter (\(i,x) -> x /= 0) paired
--[(0,1), (1,2), (4,4)]
q05_ind = map fst filtered
--fst　ペアの１番目の値を取り出す関数

--6. Create a 3x3 identity matrix (★☆☆)
--identity　単位行列
q06_matrix :: Tensor
q06_matrix = reshape [3, 3] (asTensor ([if i == j then 1 else 0 | j <- [0..2], i <- [0..2]] :: [Int]))

--7. Create a 3x3x3 array with random values (★☆☆)
q07_lst :: [Float]
q07_lst = Prelude.take 27 (randoms (mkStdGen 100))
q07_array :: Tensor
q07_array = reshape [3, 3, 3] (asTensor q07_lst)

--8. Create a 10x10 array with random values and find the minimum and maximum values (★☆☆)
q08_lst :: [Float]
q08_lst = Prelude.take 100 (randoms (mkStdGen 30))
q08_array :: Tensor
q08_array = reshape [10, 10] (asTensor q08_lst)
q08_max = maximum q08_lst
q08_min = minimum q08_lst

--9. Create a random vector of size 30 and find the mean value (★☆☆)
q09_lst :: [Float]
q09_lst = Prelude.take 30 (randoms (mkStdGen 3))
q09_mean = (sum q09_lst) / fromIntegral (length q09_lst)

--10. Create a 2d array with 1 on the border and 0 inside (★☆☆)
--行列のふちだけを１にして、内側を０にする
--5*5の行列だとする
q10_matrix :: Tensor
q10_matrix = reshape [5, 5] (asTensor ([if i==0 || i == 4 || j == 0 || j == 4 then 1 else 0 | i <- [0..4], j <- [0..4]] :: [Int]))

--11. How to add a border (filled with 0's) around an existing array? (★☆☆)
--ある行列の周りに０で埋められたフチを作る
q11_original = [[1,1,1],
                [1,1,1],
                [1,1,1]]
q11_middle = [[0] ++ row ++ [0] | row <- q11_original]
q11_zeroRow = replicate 5 0
q11_padded = [q11_zeroRow] ++ q11_middle ++ [q11_zeroRow]

--12. Create a 5x5 matrix with values 1,2,3,4 just below the diagonal (★☆☆)
q12_matrix :: Tensor
q12_matrix = reshape [5,5] (asTensor ([if i == j+1 then i else 0 | i <- [0..4], j <- [0..4]] :: [Int]))

--13. Consider a (6,7,8) shape array, what is the index (x,y,z) of the 100th element?
x = 99 `Prelude.div` (7*8) 
y = (99 - (7*8*x)) `Prelude.div` 8
z = (99 - (7*8*x)) `mod` 8
q13_ans = [x, y, z]

--14. Normalize a 5x5 random matrix (★☆☆)
q14_matrix :: Tensor
q14_matrix = reshape [5, 5] (asTensor (Prelude.take 25 (randoms (mkStdGen 28)) :: [Float]))
q14_min = Torch.min q14_matrix
q14_max = Torch.max q14_matrix
q14_normalized = (q14_matrix - q14_min) / (q14_max - q14_min)

--15. Multiply a 5x3 matrix by a 3x2 matrix (real matrix product) (★☆☆)
q15_matrix_1 = [[1,2,3],
                [1,2,3],
                [1,2,3],
                [1,2,3],
                [1,2,3]]
q15_matrix_2 = [[1,2],
                [1,2],
                [1,2]]

q15_zipWith = [zipWith (*) rowA colB | rowA <- q15_matrix_1 ,colB <- (Data.List.transpose q15_matrix_2)]
q15_sum = map sum q15_zipWith
q15_ans :: Tensor
q15_ans = reshape [5,2] (asTensor (q15_sum :: [Int]))

--16. Given a 1D array, negate all elements which are between 3 and 8, in place. (★☆☆)
q16_array = [1,4,8,2,3,7,9,6,4,7]
q16_ans = [if 3 <= x && x <= 8 then -x else x | x <- q16_array]

--17. How to round away from zero a float array ? (★☆☆)
q17_array = Prelude.take 10 (randomRs (-10, 10) (mkStdGen 6) :: [Float])
-- q17_ans = [if x > 0 && x `mod` 1 /= 0 then (x `Prelude.div` 1) + 1 else if x < 0 && x `mod` 1 /= 0 then (x `Prelude.div` 1) -1 else x | x <- q17_array]
q17_ans = [if x > 0 then fromIntegral (ceiling x) else fromIntegral (Prelude.floor x) | x <- q17_array]

--18. How to find common values between two arrays? (★☆☆)
--　common value とは、二つの行列の両方に存在している共通の数字
-- concat 行列の壁を壊して、一列の平坦なリストにする
-- intersect で二つのリストの共通する数字をあぶり出す
-- nub　もし同じ数字がダブっていたら、一つにまとめる
q18_array_1 = [[1,2,3],
               [4,5,6]]
q18_array_2 = [[5,6,7],
               [8,9,10]]
q18_flat_1 = concat q18_array_1
q18_flat_2 = concat q18_array_2
-- q18_ans = nub (intersect q18_flat_1 q18_flat_2)
q18_ans = [x | x <- q18_flat_1, y <- q18_flat_2, x==y]

--19. How to compute ((A+B)*(-A/2)) in place (without copy)? (★★☆)
-- BにA+Bを上書きし、AにA/2を上書きし、Aにマイナスをつけて上書き。最後にAにA*Bを上書きする
-- Haskellは純粋関数型言語だから、データは全て不変
q19_a = 2 
q19_b = 4
q19_ans = (q19_a + q19_b) * (-(q19_a/2))

--20. Extract the integer part of a random array of positive numbers using 4 different methods (★★☆)
-- ランダムな正の数が入った配列から、整数部分だけを抽出せよ
q20 = Prelude.take 10 (randomRs (0, 10) (mkStdGen 6) :: [Float])
-- ① 切り捨て　truncate　
q20_1 = [fromIntegral (truncate x) | x <- q20]
-- ② 切り下げ　floor
q20_2 = [fromIntegral (Prelude.floor x) | x <- q20]
-- ③ 帯分数化　properFraction　整数と少数のペアに分割してくれる
q20_3 = [fromIntegral (fst (properFraction x)) | x <- q20]
-- ④ 四捨五入　round あらかじめ0.5を引いてから四捨五入する
q20_4 = [fromIntegral (round (x - 0.5)) | x <- q20]



main :: IO ()
main = do
    
    print q01_vector
    print q02_vector
    print q03_vector
    print q04_matrix
    print q05_ind
    print q06_matrix
    print q07_array
    print q08_array
    print q08_max
    print q08_min
    print q09_lst
    print q09_mean
    print q10_matrix
    mapM_ print q11_padded
    print q12_matrix
    print q13_ans
    print q14_normalized
    print q15_ans
    print q16_ans
    print q17_array
    print q17_ans
    print q18_ans
    print q19_ans
    print q20
    print q20_1
    print q20_2
    print q20_3
    print q20_4


    