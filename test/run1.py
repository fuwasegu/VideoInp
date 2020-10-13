import subprocess
import gc

for i in range(1, 26):
    if i % 2 == 0:
        print('Start process No. ' + str(i) + ' ...')
        subprocess.call(['python', 'test_10_12.py', str(i)])
        gc.collect()
        print('-------------------')


