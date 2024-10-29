import os
fname = os.path.join("jena_climate_2009_2016.csv")
with open(fname) as f:
    data = f.read()
    lines = data.split("\n")
    header = lines[0].split(",")
    lines = lines[1:]
    print(header)
    print(len(lines))