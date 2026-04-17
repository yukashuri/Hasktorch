
module Pre (main) where

--hasktorch
import Torch.TensorFactories (zeros')

main :: IO ()
main = do
  print $ zeros' [3, 4]