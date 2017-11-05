import glob
for f in glob.glob("BURNS/*Patient*.xls"):
    if "Patient" in f:
        target = "BURNS_CSV/"+f.split("/")[-1] + "_%s.csv"
        f = f.replace(" ", "\\ ")
        target = target.replace(" ", "\\ ")
        cmd = "ssconvert %s %s -S"%(f, target)
        print cmd
