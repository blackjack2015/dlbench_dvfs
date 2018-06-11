import pandas as pd
import numpy as np

logRoot = 'excels/titanX-vgg-logs-v2'
iters = 51
baseCF = 1600
baseMF = 3500

csv_perf = "%s-segment.csv" % logRoot
df = pd.read_csv(csv_perf, header = 0)

print df.head(3)
print df.dtypes

coreF = df['coreF'].unique()
memF = df['memF'].unique()
kernels = df['kernel'].unique()
coreF.sort()
memF.sort()
kernels.sort()

print "coreF\tmemF\ttotal time(ms)\taver time(ms)"
for cF in coreF:
	for mF in memF:
		curData = df[(df['coreF'] == cF) & (df['memF'] == mF)]
		totalTime = sum(curData['count'] * curData['aver_time'])

		print "%d\t%d\t%f\t%f\t" % (cF, mF, totalTime * 1000, totalTime * 1000 / iters)

print "core sensitivity\tmemory sensitivity\tsensitivity\tkernel"
for kernel in kernels:
	curData = df[df['kernel'] == kernel]
	# print curData

	# core frequency sensitivity
	coreS = [ float((item['aver_time'] / curData[(curData['coreF'] == baseCF) & (curData['memF'] == item['memF'])]['aver_time']) / \
			  ((item['coreF']) * 1.0 / baseCF)) for idx, item in curData.iterrows() ]
	aver_coreS = np.mean(coreS)

	# memory frequency sensitivity
	memS = [ float((item['aver_time'] / curData[(curData['coreF'] == item['coreF']) & (curData['memF'] == baseMF)]['aver_time']) / \
			  ((item['memF']) * 1.0 / baseMF)) for idx, item in curData.iterrows() ]
	aver_memS = np.mean(memS)

	# core-memory frequency sensitivity
	S = [ float((item['aver_time'] / curData[(curData['coreF'] == baseCF) & (curData['memF'] == baseMF)]['aver_time']) / \
			  (item['memF'] * 1.0 / baseMF * item['coreF'] * 1.0 / baseCF)) for idx, item in curData.iterrows() ]
	aver_S = np.mean(S)

	print "%f\t%f\t%f\t%s" % (aver_coreS, aver_memS, aver_S, kernel[:10])