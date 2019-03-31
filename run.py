# import the necessary packages
from __future__ import print_function
import argparse
import app_logic as logic

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--display", type=int, default=-1,
	help="Whether or not frames should be displayed")
args = vars(ap.parse_args())

if args["display"] > 0:
	logic.display()

