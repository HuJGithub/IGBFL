import os
import sys
from utils.args_util import parse_args
from pipeline.Pipeline import Pipeline
import datetime
import time
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
# -d "motivation" -p "artificial_bug" -i any -m "GP02"  -e cc
def main():

    #start = time.time()
    project_dir = os.path.dirname(__file__)
    configs = parse_args(sys.argv)


    sys.argv = ["run.py"]
    print(11)
    pl = Pipeline(project_dir, configs)
    pl.run()


if __name__ == "__main__":
    main()
