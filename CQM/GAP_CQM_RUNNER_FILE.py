import sys
import GAP_CQM

args = sys.argv[1:]

filepath = args[0]
time_limit = int(args[1])

GAP_CQM.GAP_CQM_SOLVER(filepath, time_limit)
