import subprocess

pathtoexe = r'D:\p00hb\Documents\MAE 600\UAM1\UAM1 4.27\WindowsNoEditor\UAM1.exe'
commands = [[pathtoexe,'-RenderOffscreen'], [pathtoexe,'-RenderOffscreen']] # add as many as you want for more environments in parallel
procs = [subprocess.Popen(i) for i in commands]
for p in procs:
    p.wait() #Will start all and wait for the first one to close until all are closed.

