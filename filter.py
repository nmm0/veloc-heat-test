import sys
import os

if (len(sys.argv) != 5):
   sys.exit('Usage: <input file> <# of nodes> <problem size (MB)> <scaling> <exe_file>')

f = open(sys.argv[1], "r")
out = []
for line in f:
   for word in line.split():
      try:
         out.append(float(word))
      except ValueError:
         pass

sum = 0
for val in out:
   sum += val
sum = float(sum / 9)
f.close()

filename = "results/average_time.txt"
if os.path.exists(filename):
   append_write = "a"
else:
   append_write = "a"

f = open("results/average_time.txt", append_write)

nodes = str(sys.argv[2])
size = str(sys.argv[3])

scale = str(sys.argv[4])
if scale == "weak":
  # weak scaling
  final = "Average iteration time across 9 runs (" + nodes + " nodes, " + size + " MB per node): " + str(sum) + "\n"
elif scale == "strong":
  # strong scaling
  final = "Average iteration time across 9 runs (" + nodes + " nodes, " + size + " MB total problem size): " + str(sum) + "\n"

f.write(final)
f.close()

