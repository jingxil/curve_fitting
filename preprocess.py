import argparse
import torch
import numpy as np
from scipy import stats
import logging
from cfit import NUM_CLASSES
from cfit import NUM_POINTS
from cfit import NUM_IN_FEATURES
from cfit import IN_FEATURE_SIZE
from cfit import MAX_OUTPUT

"""
Read all data and split data into train set (80%) valid set (10%) and test set (10%).
"""


def split_xy(data):
  return data[:,:NUM_IN_FEATURES*IN_FEATURE_SIZE].reshape((-1,NUM_IN_FEATURES,IN_FEATURE_SIZE)), data[:,NUM_IN_FEATURES*IN_FEATURE_SIZE:NUM_IN_FEATURES*IN_FEATURE_SIZE+NUM_POINTS]

def cat_xy(x,y):
  return np.concatenate((x,y),axis=1)

def map2class(y):
  """ map [0,MAX_OUTPUT] to NUM_CLASSES classes 
  """
  assert np.any(y>MAX_OUTPUT)==False

  interval = MAX_OUTPUT/(NUM_CLASSES-1)
  y = np.round(y/interval)
  y = y.astype(int)
  return y


def save_as_bin(x, y, path):
  x = torch.from_numpy(x)
  y = torch.from_numpy(y)
  torch.save((x,y), path)

def main():
  logging.basicConfig(
      level=logging.DEBUG,
      format="[%(asctime)s] %(levelname)s: %(message)s"
  )
  parser = argparse.ArgumentParser(description='preprocess.py')
  parser.add_argument('-data_path', required=True)
  args = parser.parse_args()

  # Read data
  logging.info('Read data')
  np_data = np.loadtxt(open(args.data_path, "rb"), dtype=np.float32, delimiter=",")
  np.random.shuffle(np_data)

  # Split data
  logging.info('Split data')
  data_num = np_data.shape[0]
  train_idx = int(data_num/10*8)
  valid_idx = int(data_num/10*9)
  train_data = np_data[:train_idx]
  valid_data = np_data[train_idx:valid_idx]
  test_data = np_data[valid_idx:]

  # Normalize data
  train_x, train_y = split_xy(train_data)
  valid_x, valid_y = split_xy(valid_data)
  test_x, test_y = split_xy(test_data)

  mean = np.mean(train_x,axis=0)
  std = np.std(train_x,axis=0)
  mean = mean.reshape((1, NUM_IN_FEATURES,IN_FEATURE_SIZE))
  std = std.reshape((1, NUM_IN_FEATURES,IN_FEATURE_SIZE))

  train_x = (train_x - mean)/std
  valid_x = (valid_x - mean)/std
  test_x = (test_x - mean)/std

  train_y = map2class(train_y)
  valid_y = map2class(valid_y) 
  test_y = map2class(test_y)   


  logging.info('Save data')
  save_as_bin(mean,std, args.data_path+'.profile.pt' )
  save_as_bin(train_x, train_y, args.data_path+'.train.pt')
  save_as_bin(valid_x, valid_y, args.data_path+'.valid.pt')
  save_as_bin(test_x, test_y, args.data_path+'.test.pt')

if __name__ == "__main__":
  main()

