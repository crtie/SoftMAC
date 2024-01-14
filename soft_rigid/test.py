import pickle
import numpy as np
# 获取文件大小（可直接嵌入工程使用）
# input:文件路径
# output：文件大小，单位
import os
def getFileSize(filePath):
    fsize = os.path.getsize(filePath)	# 返回的是字节大小
    if fsize < 1024:
    	return(round(fsize,2),'Byte')
    else: 
    	KBX = fsize/1024
    	if KBX < 1024:
    		return(round(KBX,2),'K')
    	else:
    		MBX = KBX /1024
    		if MBX < 1024:
    			return(round(MBX,2),'M')
    		else:
    			return(round(MBX/1024),'G')


print(getFileSize('/data/partical_object/jacobian/plane/1deb997079e0b3cd6c1cd53dbc9f7b8e.pkl'))
with open('/data/partical_object/jacobian/plane/1deb997079e0b3cd6c1cd53dbc9f7b8e.pkl', 'rb') as f:
	a = pickle.load(f)
print(len(a))
print(type(a))
print(a[0].keys())
for i in range(1):
	for key in a[i].keys():
		print(key, type(a[i][key]))
		if type(a[i][key]) == np.ndarray:
			print(a[i][key].shape)
