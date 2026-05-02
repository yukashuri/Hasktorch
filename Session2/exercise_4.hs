module Main (main) where


--hasktorch
import Torch
import Torch.Functional
import System.Random
import Data.Int
import Data.Ord (comparing)
import qualified Data.Map as Map
import Data.Set as Set
import Data.List

p0 = asTensor ([ [0.0, 0.0],
                 [1.0, 2.0],
                 [3.0, 4.0]] :: [[Float]])

p1 = asTensor ([ [1.0, 0.0],
                 [2.0, 3.0],
                 [2.0, 2.0]] :: [[Float]])

p_ = asTensor ([0.5, 0.5] :: [Float])

p = asTensor ([ [0, 0],
                [0.5, 0.5],
                [1.0, 1.0]] :: [[Float]])

-- 61. Consider 2 sets of points P0,P1 describing lines (2d) and a point p, how to compute distance from p to each line i (P0[i],P1[i])? (★★★)
-- 直線(2D)を表す2つの点の集合 P0, P1 と、ある点 p があるとする。点 p から各直線 i (P0[i], P1[i]) への距離をどうやって計算するか？ (★★★)
-- vv = [((p1_x - p0_x), (p1_y - p0_y)) | [p1_x, p1_y] <- p1, [p0_x, p0_y] <- p0]
-- ww = [((p_x - p0_x), (p_y - p0_y)) | [p_x, p_y] <- p, [p0_x, p0_y] <- p0]
q61_v = p1 - p0
q61_w = p_ - p0
q61_v_x = select 1 0 q61_v
q61_v_y = select 1 1 q61_v
q61_w_x = select 1 0 q61_w
q61_w_y = select 1 1 q61_w
q61_numerator = Prelude.abs (q61_v_x * q61_w_y - q61_v_y * q61_w_x)
q61_denominator = Torch.sqrt (q61_v_x * q61_v_x + q61_v_y * q61_v_y)
q61_d = q61_numerator / q61_denominator

-- 62. Consider 2 sets of points P0,P1 describing lines (2d) and a set of points P, how to compute distance from each point j (P[j]) to each line i (P0[i],P1[i])? (★★★)
-- 直線(2D)を表す2つの点の集合 P0, P1 と、点の集合 P があるとする。各点 j (P[j]) から各直線 i (P0[i], P1[i]) への距離をどうやって計算するか？ (★★★)
-- q62_v = p1 - p0
-- q62_w = p - p0
-- q62_v_x = select 1 0 q62_v
-- q62_v_y = select 1 1 q62_v
-- q62_w_x = select 1 0 q62_w
-- q62_w_y = select 1 1 q62_w
-- q62_numerator = Prelude.abs (q62_v_x * q62_w_y - q62_v_y * q62_w_x)
-- q62_denominator = Torch.sqrt (q62_v_x * q62_v_x + q62_v_y * q62_v_y)
-- q62_d = q62_numerator / q62_denominator

-- 1. p の1番目（真ん中）の次元を膨らませて [3, 1, 2] にする！
p_expanded = unsqueeze (Dim 1) p
-- 2. 膨らませた p を使ってベクトルを作る
q62_v = p1 - p0
q62_w = p_expanded - p0
q62_v_x = select 1 0 q62_v
q62_v_y = select 1 1 q62_v
-- w の次元が1つ増えたので、XとYの次元の位置も 1つ後ろ(Dim 2) にズレます！
q62_w_x = select 2 0 q62_w
q62_w_y = select 2 1 q62_w
-- Prelude. を Torch. に変更！
q62_numerator = Torch.abs (q62_v_x * q62_w_y - q62_v_y * q62_w_x)
q62_denominator = Torch.sqrt (q62_v_x * q62_v_x + q62_v_y * q62_v_y) + 1e-6
-- 計算結果 d は [3, 3] の形（9パターン全ての距離）になります！
q62_d = q62_numerator / q62_denominator

-- 63. Consider an arbitrary array, write a function that extract a subpart with a fixed shape and centered on a given element (pad with a fill value when necessary) (★★★)
-- 任意の配列があるとする。指定された要素を中心として、固定された形（Shape）の部分配列を抽出する関数を書け（必要に応じて特定の値でパディングすること）。 (★★★)
q63 = ones' [5, 5]
-- paddingの設定: [左, 右, 上, 下] にいくつ足すか
q63_padAmount = [1, 1, 1, 1]
padTopBottom = zeros' [1, 5]
-- 2. cat で上下にくっつける（Dim 0 = 縦方向の結合）
-- 結果は [7, 5] の形になります
q63_padY = cat (Dim 0) [padTopBottom, q63, padTopBottom]
-- 3. 左右用のパディング（7行1列のゼロ）を作る
padLeftRight = zeros' [7, 1]
-- 4. cat で左右にくっつける（Dim 1 = 横方向の結合）
-- これで完璧な [7, 7] のパディング配列が完成！
q63_padded = cat (Dim 1) [padLeftRight, q63_padY, padLeftRight]
-- slice (次元) (開始位置) (終了位置) (ステップ) (テンソル)
-- Dim 0（行）を 0行目〜3行目(手前) まで切り取る
q63_sliceY = sliceDim 0 0 3 1 q63_padded
-- Dim 1（列）を 0列目〜3列目(手前) まで切り取る
q63_ans = sliceDim 1 0 3 1 q63_sliceY

-- 64. Consider an array Z = [1,2,3,4,5,6,7,8,9,10,11,12,13,14], how to generate an array R = [[1,2,3,4], [2,3,4,5], [3,4,5,6], ..., [11,12,13,14]]? (★★★)
-- 配列 Z = [1,2,3,4,5,6,7,8,9,10,11,12,13,14] があるとする。ここから配列 R = [[1,2,3,4], [2,3,4,5], [3,4,5,6], ..., [11,12,13,14]] をどうやって生成するか？ (★★★)
q64 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
q64_ans = [ x | x <- Prelude.map (Prelude.take 4) (tails q64), length x == 4]

-- 65. Compute a matrix rank (★★★)
-- 行列の階数（ランク）を計算せよ。
-- サンプル行列 m
q65 = asTensor ([[1.0, 2.0, 3.0],
                   [2.0, 4.0, 6.0],
                   [0.0, 1.0, 0.0]] :: [[Float]])

-- svd を実行すると、(u, s, v) の3つのテンソルが返ってくる
-- この真ん中の `s` が「特異値（意味の強さ）」のリスト
-- ゼロ以外の値が入っている数がランク
(u, s, v) = svd False False q65
q65_ans = s

-- 66. How to find the most frequent value in an array?
-- 配列の中で最も頻繁に出現する値（最頻値）をどうやって見つけるか？
-- q66 = asTensor ([1, 2, 2, 3, 2, 4] :: [Int])
-- q66_count = bincount q66
-- q66_ans = argmax (Dim 0) KeepDim q66


-- 77. Considering a 10x3 matrix, extract rows with unequal values (e.g. [2,2,3]) (★★★)
-- 10x3 の行列があるとする。要素がすべて同じではない行（例：[2,2,3] など、すべて同値でないもの）を抽出せよ。 (★★★)
q77 = [
            [1.0, 1.0, 1.0],  -- 1行目: 全部同じ（除外対象）
            [2.0, 2.0, 3.0],  -- 2行目: 違う値あり（抽出対象）
            [0.0, 0.0, 0.0],  -- 3行目: 全部同じ（除外対象）
            [1.0, 2.0, 1.0],  -- 4行目: 違う値あり（抽出対象）
            [4.0, 4.0, 4.0],  -- 5行目: 全部同じ（除外対象）
            [3.0, 1.0, 2.0],  -- 6行目: 違う値あり（抽出対象）
            [5.0, 5.0, 5.0],  -- 7行目: 全部同じ（除外対象）
            [0.0, 1.0, 0.0],  -- 8行目: 違う値あり（抽出対象）
            [9.0, 9.0, 9.0],  -- 9行目: 全部同じ（除外対象）
            [7.0, 7.0, 8.0]   -- 10行目: 違う値あり（抽出対象）
        ] 
q77_ans = [ [x,y,z] | [x,y,z] <- q77, x==y && y==z]



main :: IO ()
main = do

    print q61_d
    print q62_d
    print q63_ans
    print q64_ans 
    print q65_ans
    -- print q66_ans
    print q77_ans