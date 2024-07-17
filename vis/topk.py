import numpy as np
import random
from PIL import Image
from matplotlib import pyplot as plt

k = 5
n = 3

gall_img = np.load('/root/UNICORN/vis/gall_img.npy')
gall_feat = np.load('/root/UNICORN/vis/gall_feat.npy')
gall_label = np.load('/root/UNICORN/vis/gall_label.npy')
query_img = np.load('/root/UNICORN/vis/query_img.npy')
query_feat = np.load('/root/UNICORN/vis/query_feat.npy')
query_label = np.load('/root/UNICORN/vis/query_label.npy')

similarity = np.matmul(query_feat, np.transpose(gall_feat))

rank = np.argsort(similarity)[:,::-1][:,:k]
    
# plt.figure(figsize=(20,5))

sample = random.sample([i for i in range(189)], n)

for i in range(n):
    plt.subplot(n,k+1,1+i*(k+1))
    plt.imshow(query_img[sample[i]])
    plt.axis('off')
    if i == 0:
        plt.title('query')
    for j in range(k):
        plt.subplot(n,k+1,i*(k+1)+j+2)
        plt.imshow(gall_img[rank[sample[i]][j]])
        plt.axis('off')
        
        if i == 0:
            plt.title(f'rank {j+1}')
        
        ax = plt.gca()
        if query_label[sample[i]] == gall_label[rank[sample[i]][j]]:
            ax.add_patch(plt.Rectangle((0,0),144,288, fill=False, edgecolor='green', linewidth=3))
        else:
            ax.add_patch(plt.Rectangle((0,0),144,288, fill=False, edgecolor='red', linewidth=3))

plt.tight_layout()

plt.savefig('/root/UNICORN/vis/topk_img.jpg')
