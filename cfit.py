import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
import logging

NUM_IN_FEATURES = 1
IN_FEATURE_SIZE = 2
HIDDEN_FEATURE_SIZE = 3
NUM_CLASSES = 11
NUM_POINTS = 1
MAX_OUTPUT = 0.4

class Conv3Block(nn.Module):
  def __init__(self, in_channels, out_channels, dropout_rate):
    super(Conv3Block, self).__init__()
    self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
    self.bn = nn.BatchNorm1d(out_channels)
    self.activation = nn.LeakyReLU()
    self.dropout = nn.Dropout(dropout_rate)

  def forward(self, x):
    out = self.conv(x)
    out = self.bn(out)
    out = self.activation(out)
    out = self.dropout(out)
    return out


class ResidualBlock(nn.Module):
  def __init__(self, nfeatures, dropout_rate):
    super(ResidualBlock, self).__init__()
    self.conv_block = Conv3Block(nfeatures, nfeatures, dropout_rate)
    self.conv = nn.Conv1d(nfeatures, nfeatures, kernel_size=3, stride=1, padding=1)
    self.bn = nn.BatchNorm1d(nfeatures)
    self.activation = nn.LeakyReLU()
    self.dropout = nn.Dropout(dropout_rate)
        
  def forward(self, x):
    residual = x
    out = self.conv_block(x)
    out = self.conv(out)
    out = self.bn(out)
    out += residual
    out = self.activation(out)
    return out


class Estimator(nn.Module):
  def __init__(self, args):
    super(Estimator, self).__init__()
    # Prepare features layers
    self.args = args
    self.feat_proj = nn.Linear(IN_FEATURE_SIZE, HIDDEN_FEATURE_SIZE) 
    self.feat_conv = Conv3Block(NUM_IN_FEATURES, args.nfeatures, args.dropout)
    self.res_blocks = nn.ModuleList()
    for n in range(args.nlayers):
      self.res_blocks.append(ResidualBlock(args.nfeatures, args.dropout))
    # Predict layers 
    self.fc = nn.Linear(args.nfeatures*HIDDEN_FEATURE_SIZE, NUM_POINTS*NUM_CLASSES)
    self.loss = nn.CrossEntropyLoss()

  def forward(self, x):
    """
    PARAMS:
      x [batch_size, NUM_IN_FEATURES, IN_FEATURE_SIZE]
    """
    out = self.feat_proj(x) # [batch_size, NUM_IN_FEATURES, HIDDEN_FEATURE_SIZE]
    out = self.feat_conv(out) # [batch_size, nfeatures, HIDDEN_FEATURE_SIZE]
    for i in range(self.args.nlayers):
      out = self.res_blocks[i](out) # [batch_size, nfeatures, HIDDEN_FEATURE_SIZE]
    out = out.view(-1,self.args.nfeatures*HIDDEN_FEATURE_SIZE) # [batch_size, nfeatures*HIDDEN_FEATURE_SIZE]
    out = self.fc(out) # [batch_size, npoints*nclasses]
    out = out.view(-1, NUM_POINTS, NUM_CLASSES) # [batch_size, npoints, nclasses]
    return out

  def compute_loss(self, out, y):
    return self.loss(out.view(-1,NUM_CLASSES), y.view(-1))

def load_data(data_path):
  data = torch.load(data_path)
  return data[0], data[1]

  
def make_batches(data_x, data_y, batch_size, is_shuffle=True):
  assert data_x.shape[1] == NUM_IN_FEATURES
  assert data_x.shape[2] == IN_FEATURE_SIZE
  assert data_y.shape[1] == NUM_POINTS
  assert data_x.shape[0] == data_y.shape[0]
  total_size = data_x.shape[0]

  if is_shuffle:
    pick = np.random.permutation(total_size)
  else:
    pick = np.arange(total_size)
  pick = pick.astype(np.int64)
  pick = torch.from_numpy(pick)
  
  idx = 0
  while idx<total_size:
    d = pick[idx:idx+batch_size]
    yield (torch.index_select(data_x,0,d),torch.index_select(data_y,0,d))
    idx+=batch_size



def statistic(out, y):
  """
  out and y should have the same shape
  """
  out = out.numpy().astype(np.float32)/(NUM_CLASSES-1)
  y = y.numpy().astype(np.float32)/(NUM_CLASSES-1)
  # absolute difference
  diff = out - y
  diff = np.abs(diff)
  for abs_diff in [0.04, 0.08, 0.1]:
    # In the view of every record
    result = diff > abs_diff
    temp = np.sum(result,1) >= 1
    ra = 1. - 1.*np.sum(temp)/temp.size
    # In the view of every point
    pa = 1. - 1.*np.sum(result)/result.size
    logging.info("Absolute difference within %.2f Record Accuracy %.6f Point Accuracy %.6f"%(abs_diff, ra, pa))

  # relative difference
  for relative_diff_rate in [0.1, 0.2]:
    relative_diff = relative_diff_rate*y
    # In the view of every record
    result = diff > relative_diff
    temp = np.sum(result,1) >= 1
    ra = 1. - 1.*np.sum(temp)/temp.size
    # In the view of every point
    pa = 1. - 1.*np.sum(result)/result.size
    logging.info("Relative difference within %.2f Record Accuracy %.6f Point Accuracy %.6f"%(relative_diff_rate, ra, pa))


def train(epoch, model, data, data_size, optimizer, args):
  model.train()
  train_loss = 0
  for batch_idx, (X, y) in enumerate(data):
    batch_size = y.size(0)
    X = Variable(X) # [batch_size, NUM_IN_FEATURES, IN_FEATURE_SIZE]
    y = Variable(y) # [batch_size, npoints]
    if args.gpu:
      X = X.cuda()
      y = y.cuda()
    optimizer.zero_grad()
    out = model(X) # [batch_size, npoints, nclasses]
    loss = model.compute_loss(out, y)
    loss.backward()
    loss = loss.data[0]
    train_loss += loss*batch_size
    optimizer.step()
    if batch_idx % 10000==0:
      logging.info("Batch [%d / %d]  loss: %.6f"%(batch_idx, data_size//batch_size, loss))
  train_loss /= data_size
  logging.info("Epoch %d loss: %.6f"%(epoch, train_loss))


def eval(model, data, data_size, args):
    model.eval()
    test_loss = 0
    pred_y = []
    true_y = []
    
    for batch_idx, (X, y) in enumerate(data):
      batch_size = y.size(0)
      X = Variable(X, volatile=True)
      y = Variable(y, volatile=True) # [batch_size, npoints]
      if args.gpu:
        X = X.cuda()
        y = y.cuda()
      out = model(X) # [batch_size, npoints, nclasses]
      loss = model.compute_loss(out, y)
      test_loss += loss.data[0]*batch_size
      # Get prediction
      _, out = torch.topk(out,1)
      out = out.squeeze(2)
      # Fetch data
      if args.gpu:
        out_ = out.data.cpu()
        y_ = y.data.cpu()
      else:
        out_ = out.data
        y_ = y.data
      pred_y.append(out_)
      true_y.append(y_)

    test_loss /= data_size
    logging.info("Test loss: %.6f" % (test_loss))
    pred_y = torch.cat(pred_y,0)
    true_y = torch.cat(true_y,0)
    return test_loss, pred_y, true_y







