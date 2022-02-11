# ---- This is <run_app.py> ----
"""
Script for running the data labeling application from the command line or inside ipython

@author: Catherine Taelman
"""

import segment_and_label
import argparse
from loguru import logger

# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #

def make_parser():

  p = argparse.ArgumentParser(
    formatter_class=argparse.RawDescriptionHelpFormatter,
    description='Run data labeling tool',
  )

  p.add_argument(
    'feat_folder',
    help='path to feature folder'
  )
  p.add_argument(
    'result_folder',
    help='path to result folder where label mask will be saved'
  )
  p.add_argument(
    '-goal',
    default='icetypes',
    help = "Goal of the labeling. Choose from 'icetypes' or 'ice-water' (default='icetypes')."
  )
  p.add_argument(
    '-loglevel',
    choices=['CRITICAL', 'ERROR', 'WARNING', 'INFO', 'DEBUG'],
    default='INFO',
    help='set logger level (default=INFO)'
  )

  return p

# -------------------------------------------------------------------------- #

if __name__ == '__main__':

  p = make_parser()
  args = p.parse_args()

  try:
    segment_and_label.main(**vars(args))
  except Exception as E:
    logger.critical(E)

# -------------------------------------------------------------------------- #

# ---- End of <run_app.py> ----
