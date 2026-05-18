{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE StandaloneDeriving #-}

module Word2Vec where
import Codec.Binary.UTF8.String (encode) -- add utf8-string to dependencies in package.yaml
import GHC.Generics
import qualified Data.ByteString.Lazy as B -- add bytestring to dependencies in package.yaml
import qualified Data.ByteString.Lazy.Char8 as BC                  -- 文字を扱いやすくするためのモジュール
import Data.Char (toLower)                                         -- 小文字化をするための関数
import Data.Word (Word8)
import qualified Data.Map.Strict as M -- add containers to dependencies in package.yaml
import Data.List (nub)

import Control.Monad (when)
import Torch
import Torch.Serialize (saveParams, loadParams)

-- import Torch.Autograd (makeIndependent, toDependent)
-- import Torch.Functional (embedding')
-- import Torch.NN (Parameterized(..), Parameter)
-- import Torch.Serialize (saveParams, loadParams)
-- import Torch.Tensor (Tensor, asTensor)
-- import Torch.TensorFactories (eye', zeros')

-- your text data (try small data first)
textFilePath = "Session6/data/sample.txt"
modelPath =  "Session6/data/sample_embedding.params"
wordLstPath = "Session6/data/sample_wordlst.txt"

data EmbeddingSpec = EmbeddingSpec {
  wordNum :: Int, -- the number of words
  wordDim :: Int  -- the dimention of word embeddings
} deriving (Show, Eq, Generic)


data Embedding = Embedding {
    wordEmbedding :: Parameter
  } deriving (Show, Generic, Parameterized)

-- Probably you should include model and Embedding in the same data class.
data Model = Model {
    -- mlp :: MLP,
    embeddings :: Embedding,
    linearLayer :: Linear          -- linear層
  } deriving (Show, Generic, Parameterized)

isUnncessaryChar :: 
  Word8 ->
  Bool
isUnncessaryChar str = str `elem` (map (head . encode)) [".", "!", "?", ",", "\"", "(", ")", "-", "_", ":", ";", "/", "$"] -- add more if necessary

preprocess ::
  B.ByteString -> -- input
  [[B.ByteString]]  -- wordlist per line
preprocess texts = map (B.split (head $ encode " ")) textLines
  where
    -- ① 全てのテキストを小文字化
    loweeTexts = BC.map toLower texts
    -- ② 小文字になったテキストから、記号を取り除く
    filteredtexts = B.pack $ filter (not . isUnncessaryChar) (B.unpack loweeTexts)
    -- ③ 改行で分割する
    textLines = B.split (head $ encode "\n") filteredtexts

wordToIndexFactory ::
  [B.ByteString] ->     -- wordlist
  (B.ByteString -> Int) -- function converting bytestring to index (unknown word: 0)
wordToIndexFactory wordlst wrd = M.findWithDefault (length wordlst) wrd (M.fromList (zip wordlst [0.. length wordlst]))

toyEmbedding ::
  EmbeddingSpec ->
  Tensor           -- embedding
toyEmbedding EmbeddingSpec{..} = 
  eye' wordNum wordDim


main :: IO ()
main = do
  -- load text file
  texts <- B.readFile textFilePath

  -- Create a unique word list
  let wordLines = preprocess texts
      wordlst = nub $ concat wordLines
      wordToIndex = wordToIndexFactory wordlst
  print wordlst

  -- Create initial embedding (wordDim × wordNum)
  let embsddingSpec = EmbeddingSpec {wordNum = length wordlst + 1, wordDim = 9}
  wordEmb <- makeIndependent $ toyEmbedding embsddingSpec
  let emb = Embedding {wordEmbedding = wordEmb}
  -- -- Linear層の初期化
  -- -- 入力サイズはwordDim（９）、出力サイズはwordNum（辞書の単語数）にする。
  -- initLinear <- sample $ LinearSpec { in_features = 9, out_features = length wordlst + 1}

  -- -- let emb = Model {
  --   embeddings = Embedding { wordEmbedding = wordEmb },
  --   linearLayer = initLinear
  -- }

  let sampleTxt = B.pack $ encode "This is awesome.\nmodel is developing"
  -- convert word to index
      idxes = map (map wordToIndex) (preprocess sampleTxt)
  -- convert to embedding
      embTxt = embedding' (toDependent $ wordEmbedding emb) (asTensor idxes)


  -- let sampleTxt = B.pack $ encode "this is awesome\nmodel is developing"
  --     -- 単語をインデックスに変換し、1つの平坦なリスト(配列)にする
  --     flatIdxes = concat $ map (map wordToIndex) (preprocess sampleTxt)
      
  --     inIdxes = init flatIdxes
  --     tgtIdxes = tail flatIdxes
      
  --     -- ★究極のハック：一度 Float のリストとして安全にテンソル化し、Int64（整数）にキャストする！
  --     inTensor  = toDType Int64 $ asTensor (map fromIntegral inIdxes :: [Float])
  --     tgtTensor = toDType Int64 $ asTensor (map fromIntegral tgtIdxes :: [Float])

  -- -- 学習ループ (500回繰り返す)
  -- let numIters = 500
  -- trainedEmb <- foldLoop emb numIters $ \state i -> do
      
  --     -- ① 順伝播 (予測スコアの計算)
  --     let embTxt = embedding' (toDependent $ wordEmbedding (embeddings state)) inTensor
  --         output = linear (linearLayer state) embTxt
          
  --     -- ② 誤差の計算 (予測と正解のズレを計算)
  --         loss = nllLoss' (logSoftmax (Dim 1) output) tgtTensor
          
  --     -- 100回ごとに画面に誤差(Loss)を表示
  --     when (i `mod` 100 == 0) $ do
  --         putStrLn $ "Iteration: " ++ show i ++ " | Loss: " ++ show loss
          
  --     -- ③ パラメーターの更新
  --     (newState, _) <- runStep state GD loss 5e-2  -- 学習率 0.05
      
  --     return newState

  -- putStrLn "Training Completed!"

 -- TODO: Train model. After training, we can obtain the trained patameter, embeddings. This is the trained embedding.
  
  -- Save params to use trained parameter in the next session
  -- trainedEmb :: Embedding
  -- saveParams trainedEmb modelPath
  -- Save word list
  B.writeFile wordLstPath (B.intercalate (B.pack $ encode "\n") wordlst)
  
  -- Load params
  -- initWordEmb <- makeIndependent $ zeros' [1]
  -- let initEmb = Embedding {wordEmbedding = initWordEmb}
  -- loadedEmb <- loadParams initEmb modelPath

  return ()