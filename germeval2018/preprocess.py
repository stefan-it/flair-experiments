import sys

filename = sys.argv[1]

with open(filename, 'rt') as f:
  lines = [line.rstrip() for line in f.readlines()]

for line in lines:

  new_line = [token.replace('#', '') for token in line.split() if not token[0] in ['@', '&', '|']]

  print(" ".join(new_line))
