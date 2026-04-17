module Main (main) where


--hasktorch
import Torch
import System.Random
import Data.Int
import Data.Ord (comparing)
import qualified Data.Map as Map
import Data.Set as Set
import Data.List


-- 41. Subtract the mean of each row of a matrix (★★☆)
-- 行列の各行について、その行の平均値を（各要素から）引きなさい。
q41 = [[1,2,3],
       [4,5,6],
       [7,8,9]]
q41_ans = [[x - (sum row / realToFrac (length row ))| x <- row] | row <- q41]

-- 42. How to sort an array by the nth column? (★★☆)
-- 配列を「n番目の列」を基準にして並べ替える（ソートする）には？
q42_matrix = [[1,5,3],
              [4,2,6],
              [7,8,9]] :: [[Int]]
q42_n = 1 :: Int
q42_ans = sortBy (comparing (!! q42_n)) q42_matrix
-- q42_nを取り出す

-- 43. How to tell if a given 2D array has null columns? (★★☆)
-- 与えられた2次元配列に「すべて0の列（または空の列）」があるかどうかを判定するには？
q43 = [[1,2],
       [0,0],
       [3,4]]
q43_ans = [0,0] `elem` q43

-- 44. Find the nearest value from a given value in an array (★★☆)
-- 配列の中から、ある特定の値に「最も近い値」を見つけなさい。
q44 = Prelude.take 10 (randomRs (0, 100) (mkStdGen 0) :: [Int])
n = 10
q44_ans = snd (minimum[(Prelude.abs (x - n), x) | x <- q44])

-- 45. Considering two arrays with shape (1,3) and (3,1), how to compute their sum using an iterator? (★★☆)
-- (1,3)と(3,1)の形の2つの配列について、イテレータ（繰り返し処理）を使ってその和を計算するには？
-- ヨコ(1, 3)とタテ(3, 1)のデータ
q45_row = [1, 2, 3] :: [Int]
q45_col = [10, 20, 30] :: [Int]
-- リスト内包表記で、すべての組み合わせを足し合わせる！
q45_ans = [ [y + x | x <- q45_row] | y <- q45_col ]

-- 46. Create an array class that has a name attribute (★★☆)
-- 名前（name）」という属性を持った配列クラスを作成せよ。
data Name a = Name {
    name :: String,
    array :: [a]
} deriving (Show)
q46_ans = Name {
    name = "Yuka",
    array = [1, 2, 3, 4] :: [Int]
}

-- 47. Consider a given vector, how to add 1 to each element indexed by a second vector (be careful with repeated indices)? (★★★)
-- あるベクトルに対し、別のベクトルで指定されたインデックス位置の要素に1を足しなさい
q47 = Prelude.take 10 (randomRs (0, 100) (mkStdGen 90) :: [Int])
q47_n = 5
q47_ans = [if x == q47_n then y+1 else y | (x, y) <- zip [0..9] q47 ]

-- 48. How to accumulate elements of a vector (X) to an array (F) based on an index list (I)? (★★★)
-- インデックスリスト(I)に基づいて、ベクトル(X)の要素を配列(F)に蓄積（合計）しなさい。
q48_i = [0, 1, 0, 2, 1, 0]
q48_x = [10, 20, 30, 40, 50, 60]
q48_ans = Map.fromListWith (+) (zip q48_i q48_x)

-- 49. Considering a (w,h,3) image of (dtype=ubyte), compute the number of unique colors (★★☆)
-- 幅w、高さh、3色（RGB）の画像データから、使われている「ユニークな色（重複を除いた色の数）」を計算せよ。
q_image = asTensor ([
                        [[255, 0, 0], [0, 255, 0]],
                        [[0, 0, 255], [255, 0, 0]] ] :: [[[Float]]] )
q49 = asValue q_image :: [[[Float]]]
q49_flat = concat q49
q49_ans = Set.size (Set.fromList q49_flat)

-- 50. Considering a four dimensions array, how to get sum over the last two axis at once? (★★★)
-- 4次元配列において、最後の2つの軸を一度に合計するには？
-- q50 を「形が [2, 3, 4, 5] で、中身がすべて 1.0 の4次元テンソル」として自動生成する！
q50 = Torch.ones' [2, 3, 4, 5]
-- 1. まず一番最後（第3軸）を潰す
q50_sum1 = sumDim (Dim 3) RemoveDim Float q50
-- 2. 次に新しい最後（元の第2軸）を潰す
q50_sum2 = sumDim (Dim 2) RemoveDim Float q50_sum1

-- 51. Considering a one-dimensional vector D, how to compute means of subsets of D using a vector S of same size describing subset indices? (★★★)
-- 1次元ベクトルDについて、グループ分けを表すベクトルSを使って、グループごとの平均を計算せよ。
q51_s = [0, 1, 0, 2, 1, 0, 1, 2, 1, 0, 2, 1, 1, 1, 2, 1, 0, 2, 0, 2]
q51_d = Prelude.take 20 (randomRs (0, 100) (mkStdGen 90) :: [Int])
q51_pairs = [ (s, (d, 1)) | (s, d) <- zip q51_s q51_d ]
q51_sumAndCount = Map.fromListWith (\(s1, c1) (s2, c2) -> (s1 + s2, c1 + c2)) q51_pairs
q51_ans = Map.map (\(total, count) -> fromIntegral total / fromIntegral count) q51_sumAndCount

-- 52. How to get the diagonal of a dot product? (★★★)
-- 行列のドット積（行列掛け算）の結果の「対角成分」だけを取得するには？
q52_matrix_1 = [[1,2,3],
                [4,5,6],
                [7,8,9]]
q52_matrix_2 = [[9,8,7],
                [6,5,4],
                [3,2,1]]

q52_zipWith = [zipWith (*) rowA colB | rowA <- q52_matrix_1 ,colB <- (Data.List.transpose q52_matrix_2)]
q52_sum = Prelude.map sum q52_zipWith
q52_pro :: Tensor
q52_pro = reshape [3,3] (asTensor (q52_sum :: [Int]))
q52_ans = diag (Diag 0) q52_pro

-- 53. Consider the vector [1, 2, 3, 4, 5], how to build a new vector with 3 consecutive zeros interleaved between each value? (★★★)
-- ベクトル [1, 2, 3, 4, 5] の各要素の間に、3つの連続した0を挟み込んだ新しいベクトルを作りなさい。
q53_sub = [[x, 0, 0, 0] | x <- [1..4]]
q53_ans = concat q53_sub ++ [5]

-- 54. Consider an array of dimension (5,5,3), how to mulitply it by an array with dimensions (5,5)? (★★★)
-- (5,5,3)の配列に、(5,5)の配列を掛け算するには？
-- 1. [5,5,3] の3次元テンソル A を用意
a = ones' [5, 5, 3]
-- 2. [5,5] の2次元テンソル B を用意
b = ones' [5, 5]
-- 3. B の形を [5, 5, 1] に「膨らませる」（ここが重要！）
-- unsqueeze 2 は「2番目の軸（最後）に新しい次元を追加して」という意味です
b_expanded = unsqueeze (Dim 2) b
-- 4. 普通に掛け算する
-- [5,5,3] * [5,5,1] は、Hasktorchが自動で [5,5,3] * [5,5,3] として計算してくれます
q54_ans = a * b_expanded

-- 55. How to swap two rows of an array? (★★★)
-- 配列の「2つの行」を入れ替える（スワップする）には？
-- 行を入れ替える関数
swapRowList :: Int -> Int -> [a] -> [a]
swapRowList i j xs = 
    [ if k == i then xs !! j
      else if k == j then xs !! i
      else x
    | (k, x) <- zip [0..] xs]
q55 = [[1,2,3],
       [4,5,6],
       [7,8,9]]
q55_ans = swapRowList 0 2 q55

-- 56. Consider a set of 10 triplets describing 10 triangles (with shared vertices), find the set of unique line segments composing all the triangles (★★★)
-- 10個の三角形を表すデータ（頂点の組み合わせ）から、それらを構成する「重複のない辺（エッジ）」のリストを見つけなさい。
-- 1つの三角形 (A, B, C) から3つの辺を作成し、(小, 大) の順に正規化する関数
getEdges :: (Int, Int, Int) -> [(Int, Int)]
getEdges (a, b, c) = [makeEdge a b, makeEdge b c, makeEdge c a]
  where
    -- 常に (小さい数字, 大きい数字) のペアにする
    makeEdge x y = if x < y then (x, y) else (y, x)

-- 複数の三角形のリストから、重複のない辺のリストを取得する関数
uniqueEdges :: [(Int, Int, Int)] -> [(Int, Int)]
uniqueEdges triangles = Set.toList (Set.fromList allEdges)
  where
    -- concatMap で全三角形の辺を抽出し、1つの平坦なリストにする
    allEdges = concatMap getEdges triangles
q56 = [(1,2,3),
       (2,3,4),
       (1,3,4)]
q56_ans = uniqueEdges q56

-- 57. Given a sorted array C that corresponds to a bincount, how to produce an array A such that np.bincount(A) == C? (★★★)
-- 各数字の出現回数（bincount）を表す配列Cから、その出現回数になるような元の配列Aを復元しなさい。
q57_c = [0, 3, 1, 0, 2]
q57_zip = zip [0..] q57_c
q57_ans = concat [replicate n x | (x, n) <- q57_zip]

-- 58. How to compute averages using a sliding window over an array? (★★★)
-- 配列に対して「スライディングウィンドウ（一定の範囲をずらしながら）」で移動平均を計算するには？
-- スライディングウィンドウで移動平均を計算する関数
-- n: ウィンドウサイズ, xs: 元の配列（リスト）
movingAverage :: Int -> [Double] -> [Double]
movingAverage n xs =
  [ sum w / fromIntegral n 
  | w <- Prelude.map (Prelude.take n) (tails xs) -- 窓を少しずつズラしてn個ずつ取得
  , length w == n                -- 要素数が足りない末尾のゴミを弾く
  ]
q58 = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
q58_w = 3
q58_ans = movingAverage q58_w q58

-- 59. Consider a one-dimensional array Z, build a two-dimensional array whose first row is (Z[0],Z[1],Z[2]) and each subsequent row is shifted by 1 (last row should be (Z[-3],Z[-2],Z[-1])) (★★★)
-- 1次元配列Zから、「3つずつの窓」を1つずつずらしながら並べた2次元配列を作りなさい。
build2DArray :: [a] -> [[a]]
build2DArray xs =
    [ w | w <- Prelude.map (Prelude.take 3) (tails xs), length w == 3 ]
q59 = [1,2,3,4,5,6]
q59_ans = build2DArray q59

-- 60. How to negate a boolean, or to change the sign of a float inplace? (★★★)
-- 真偽値（Boolean）を反転させたり、小数の符号を「その場で（inplace）」反転させるには？
-- Haskellは元のデータを直接書き換えることができないから新しいリストを作ることになる。
q60_Bool = [True, False, True, False]
q60_Bool_ans = Prelude.map not q60_Bool
q60_floats = [-1.5, 3.2, -5.2, 4.0]
q60_floats_ans = Prelude.map negate q60_floats


main :: IO ()
main = do
    
    print q41_ans
    print q42_ans 
    print q43_ans
    print q44_ans
    print q45_ans
    print q46_ans
    print q47
    print q47_ans
    print q48_ans
    print q49_ans
    print q50_sum2
    print q51_ans
    print q52_ans
    print q53_ans
    print q54_ans
    print q55_ans
    print q56_ans
    print q57_ans
    print q58_ans
    print q59_ans
    print q60_Bool_ans
    print q60_floats_ans