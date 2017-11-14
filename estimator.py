import argparse
import torch
import logging

import cfit
from cfit import Conv3Block
from cfit import ResidualBlock
from cfit import Estimator

def parse_args():
  logging.basicConfig(
      level=logging.DEBUG,
      format="[%(asctime)s]: %(levelname)s: %(message)s"
  )
  parser = argparse.ArgumentParser(description='estimator.py')
  parser.add_argument('-model', default='./models')
  parser.add_argument('-data', default='data.txt.test.pt')
  parser.add_argument('-batch_size', type=int, default=64)
  parser.add_argument('-gpu', action='store_true', default=False)
  parser.add_argument('-verbose', action='store_true', default=False)
  args = parser.parse_args()
  return args

def tensor2txt(tensor):
  tensor = tensor.numpy()
  txt = ''
  for t in tensor:
    txt+= ','.join(['%d'%(i) for i in t])+'\n'
  return txt

def main():
  args = parse_args()
  args.gpu = args.gpu and torch.cuda.is_available()

  gold_f = open('gold.txt','w')
  pred_f = open('pred.txt','w')

  test_x, test_y = cfit.load_data(args.data)
  model = torch.load(args.model)
  if args.gpu:
    model = model.cuda()
  else:
    model = model.cpu()

  batched_data = cfit.make_batches(test_x, test_y, args.batch_size, is_shuffle=False)
  loss, pred_y, y = cfit.eval(model, batched_data, test_x.size(0), args)
  cfit.statistic(pred_y, y)

  gold_f.write(tensor2txt(y))
  pred_f.write(tensor2txt(pred_y))

  gold_f.close()
  pred_f.close()


if __name__ == '__main__':
  main()
