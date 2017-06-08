from builtins import range
import sys
sys.path.insert(1,"../../../")
import h2o
from tests import pyunit_utils
from h2o.estimators.random_forest import H2ORandomForestEstimator

def test_data_with_weights():

  train  = h2o.upload_file(pyunit_utils.locate("smalldata/airlines/modified_airlines.csv"))
  splits = train.split_frame(ratios=[0.7])
  train = splits[0]
  test = splits[1]

  hh1 = H2ORandomForestEstimator(ntrees=1000, seed=1234)


  hh1.train(x=range(10), y=31, training_frame=train, validation_frame=test, weights_column="weight",
            score_tree_interval=10, stopping_rounds=20, stopping_metric="AUC", stopping_tolerance=0.001,
            max_runtime_secs=20*60)


if __name__ == "__main__":
  pyunit_utils.standalone_test(test_data_with_weights)
else:
  test_data_with_weights()
