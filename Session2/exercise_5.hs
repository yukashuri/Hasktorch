
-- 標準ライブラリ
import Prelude hiding (sum, max, min, div) -- 衝突を避けるため一部隠す
import qualified Prelude as P

-- Torch 関連
import Torch
import Torch.Tensor (asTensor, asValue, Tensor, toType)
import qualified Torch.Functional as F
import qualified Torch.NN as NN
import qualified Torch.Optim as Optim

-- デバイス・型
import Torch.Device (Device(..), DeviceType(..))
import Torch.DType (DType(..))

-- リスト操作・乱数
import qualified Data.List as L
import System.Random (randomRIO)
import Control.Monad (replicateM)

--67
extract3x3 :: [[a]] -> [[[a]]]
extract3x3 matrix =
  [ [ Prelude.take 3 (drop c row) | row <- Prelude.take 3 (drop r matrix) ]
  | r <- [0..7]  -- 行の開始インデックス (10 - 3 = 7 なので 0〜7)
  , c <- [0..7]  -- 列の開始インデックス
  ]

main :: IO ()
main = do

    -- 67. Extract all the contiguous 3x3 blocks from a random 10x10 matrix (★★★)
    -- ランダムな 10x10 の行列から、隣接するすべての 3x3 のブロックを抽出せよ。 (★★★)
    -- テスト用の 10x10 行列を作成 (1〜10, 11〜20... と数字が並ぶ)
    let matrix = [ [row * 10 + col | col <- [1..10]] | row <- [0..9] ]
    -- 抽出して最初の2つのブロックだけ出力してみる
    let blocks = extract3x3 matrix
    putStrLn "--- 1つ目のブロック (r=0, c=0) ---"
    print $ blocks !! 0
    putStrLn "--- 2つ目のブロック (r=0, c=1) ---"
    print $ blocks !! 1

    -- 68. Create a 2D array subclass such that Z[i,j] == Z[j,i] (★★★)
    -- Z[i,j] == Z[j,i] となるような（つまり対称行列となる）2次元配列のサブクラスを作成せよ。 (★★★)

    -- 69. Consider a set of p matrices wich shape (n,n) and a set of p vectors with shape (n,1). How to compute the sum of of the p matrix products at once? (result has shape (n,1)) (★★★)
    -- 形が (n,n) の p 個の行列の集合と、形が (n,1) の p 個のベクトルの集合があるとする。これら p 個の行列積の和を一度に計算するにはどうすればよいか？（結果の形は (n,1) になる） (★★★)

    -- 70. Consider a 16x16 array, how to get the block-sum (block size is 4x4)? (★★★)
    -- 16x16 の配列があるとする。ブロックサイズを 4x4 とした場合のブロックごとの合計値（block-sum）をどうやって取得するか？ (★★★)

    -- 71. How to implement the Game of Life using numpy arrays? (★★★)
    -- NumPy配列（テンソル）を使って「ライフゲーム（Conway's Game of Life）」をどうやって実装するか？ (★★★)

    -- 72. How to get the n largest values of an array (★★★)
    -- 配列の中から、上位 n 個の大きな値をどうやって取得するか？ (★★★)

    -- 73. Given an arbitrary number of vectors, build the cartesian product (every combinations of every item) (★★★)
    -- 任意の数のベクトルが与えられたとき、それらの直積（すべての要素のすべての組み合わせ）を構築せよ。 (★★★)
    -- 1. 任意の数のベクトルの定義 (リストのリスト)
    let v1 = [1, 2] :: [Float]
        v2 = [3, 4] :: [Float]
        v3 = [5]    :: [Float]
        allVectors = [v1, v2, v3]

    -- 2. デカルト積の構築
    -- sequence 関数は [[a]] -> [[a]] の型を持ち、
    -- 全ての組み合わせ（デカルト積）を生成します。
    -- これは [ [x,y,z] | x <- v1, y <- v2, z <- v3 ] と同等です。
    let cartesianList = sequence allVectors

    -- 3. 結果をテンソルに変換
    let q73_ans = asTensor cartesianList

    putStrLn "=== Input Vectors ==="
    P.print v1
    P.print v2
    P.print v3

    putStrLn "\n=== Cartesian Product (Every Combination) ==="
    -- 形状は (組み合わせ数, ベクトルの数) になります
    print q73_ans

    

    -- 74. How to create a record array from a regular array? (★★★)
    -- 通常の配列から、レコード配列（Record array / 構造化配列）をどうやって作成するか？ (★★★)
    

    -- 75. Consider a large vector Z, compute Z to the power of 3 using 3 different methods (★★★)
    -- 巨大なベクトル Z があるとする。Z の3乗を、3つの異なる方法を使って計算せよ。 (★★★)

    -- 76. Consider two arrays A and B of shape (8,3) and (2,2). How to find rows of A that contain elements of each row of B regardless of the order of the elements in B? (★★★)
    -- 形が (8,3) の配列 A と、形が (2,2) の配列 B があるとする。B の各行の要素の順番に関わらず、B の各行の要素を含んでいる A の行をどうやって見つけるか？ (★★★)

    -- 78. Convert a vector of ints into a matrix binary representation (★★★)
    -- 整数のベクトルを、行列の2進数表現（バイナリ表現）に変換せよ。 (★★★)
    let vList = [5, 3, 0, 7] :: [Int]
        v = toType Int32 (asTensor vList)
        numBits = 4 :: Int

    -- 2^i の重みベクトルを作成
    let powers = asTensor (map (2^) [0..numBits-1] :: [Int])
        vExpanded = reshape [head (shape v), 1] v
        
        -- 各要素を 2^j で割った商の行列
        -- Torch.div は確実にスコープにある整数除算関数です
        d = Torch.div vExpanded powers
        
        -- 剰余演算 (d mod 2) を標準演算子のみで定義
        -- result = d - ( (d / 2) * 2 ) と同等
        two = asTensor (2 :: Int)
        q78_ans = d - (Torch.div d two * two)

    putStrLn "=== Input Vector ==="
    print v

    putStrLn "\n=== Matrix Binary Representation ==="
    print q78_ans

    -- 79. Given a two dimensional array, how to extract unique rows? (★★★)
    -- 2次元配列が与えられたとき、重複しない一意な行（ユニークな行）をどうやって抽出するか？ (★★★)

    -- 80. Considering 2 vectors A & B, write the einsum equivalent of inner, outer, sum, and mul function (★★★)
    -- 2つのベクトル A と B があるとする。inner（内積）、outer（外積）、sum（合計）、mul（要素ごとの積）の関数と同等の処理を、einsum（アインシュタインの縮約記法）を使って書け。 (★★★)

    -- 81. Considering a path described by two vectors (X,Y), how to sample it using equidistant samples (★★★)?
    -- 2つのベクトル (X,Y) で表される経路があるとする。これを等間隔でサンプリングするにはどうすればよいか？ (★★★)

    -- 82. Given an integer n and a 2D array X, select from X the rows which can be interpreted as draws from a multinomial distribution with n degrees, i.e., the rows which only contain integers and which sum to n. (★★★)
    -- 整数 n と2次元配列 X が与えられたとき、n 回の試行を持つ多項分布からの抽出と解釈できる行を X から選択せよ。つまり、整数のみを含み、かつ要素の合計が n になる行を選択すること。 (★★★)

    -- 83. Compute bootstrapped 95% confidence intervals for the mean of a 1D array X (i.e., resample the elements of an array with replacement N times, compute the mean of each sample, and then compute percentiles over the means). (★★★)
    -- 1次元配列 X の平均に対する、ブートストラップ法を用いた95%信頼区間を計算せよ（つまり、配列の要素を復元抽出で N 回リサンプリングし、各サンプルの平均を計算した上で、それら平均値のパーセンタイルを計算すること）。 (★★★)
