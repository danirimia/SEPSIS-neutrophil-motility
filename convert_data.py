import glob
for f in glob.glob("TRACKS4/Spots*.xls"):
    target = "csvs/"+f.split("/")[-1] + ".csv"
    f = f.replace(" ", "\\ ")
    if "(" in f:
        continue
    #f = f.replace("(", "\(")
    #f = f.replace(")", "\)")
    target = target.replace(" ", "\\ ")
    cmd = "ssconvert %s %s"%(f, target)
    print cmd
