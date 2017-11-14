import logging
import argparse
import torch
import os
from torch import optim

import cfit

def parse_args():
  logging.basicConfig(
      level=logging.DEBUG,
      format="[%(asctime)s]: %(levelname)s: %(message)s"
  )

  parser = argparse.ArgumentParser(description='trainer.py')
  parser.add_argument('-train_data', default='data.pt')
  parser.add_argument('-valid_data', default='data.pt')
  parser.add_argument('-batch_size', type=int, default=128)
  parser.add_argument('-lr', type=float, default=0.001)
  parser.add_argument('-epochs', type=int, default=2)
  parser.add_argument('-gpu', action='store_true', default=False)
  parser.add_argument('-save_model', default='./models')
  parser.add_argument('-verbose', action='store_true', default=False)
  parser.add_argument('-dropout', type=float, default=0.01)
  parser.add_argument('-nfeatures', type=int, default=32)
  parser.add_argument('-nlayers', type=int, default=1)
  args = parser.parse_args()
  return args

def main():
  args = parse_args()
  args.gpu = args.gpu and torch.cuda.is_available()
  # make sure dir exist
  if not os.path.isdir(args.save_model):
    os.mkdir(args.save_model)
  logging.info("Load data")
  train_x, train_y = cfit.load_data(args.train_data)
  valid_x, valid_y =  cfit.load_data(args.valid_data)
  train_data_size = train_x.size(0)
  eval_data_size = valid_x.size(0)
  model = cfit.Estimator(args)
  if args.gpu:
    model.cuda()
  optimizer = optim.Adam(model.parameters(), lr=args.lr)
  for epoch in range(1, args.epochs + 1):
    # Train
    logging.info("Shuffle data")
    batched_train_data = cfit.make_batches(train_x, train_y, args.batch_size, is_shuffle=True)
    logging.info("Start training")
    cfit.train(epoch, model, batched_train_data, train_data_size, optimizer, args)
    # Eval
    batched_eval_data = cfit.make_batches(valid_x, valid_y, args.batch_size, is_shuffle=False)
    eval_loss,pred_y,y = cfit.eval(model, batched_eval_data, eval_data_size, args)
    cfit.statistic(pred_y, y)
    # Save model  
    torch.save(model, args.save_model+'/model_%0.6f_e%d.pt'%(eval_loss, epoch))

if __name__ == "__main__":
  main()
