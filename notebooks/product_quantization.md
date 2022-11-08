---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.14.1
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

```{code-cell} ipython3
import os
os.chdir('/workspace')
```

```{code-cell} ipython3
import torch
from utils.utils import kmeans,square_distance
```

# Lets Get Some Data

```{code-cell} ipython3

import shutil
import urllib.request as request
from contextlib import closing

# first we download the Sift1M dataset
with closing(request.urlopen('ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz')) as r:
    with open('sift.tar.gz', 'wb') as f:
        shutil.copyfileobj(r, f)
        
        
import tarfile

# the download leaves us with a tar.gz file, we unzip it
tar = tarfile.open('sift.tar.gz', "r:gz")
tar.extractall()


import numpy as np

# now define a function to read the fvecs file format of Sift1M dataset
def read_fvecs(fp):
    a = np.fromfile(fp, dtype='int32')
    d = a[0]
    return a.reshape(-1, d + 1)[:, 1:].copy().view('float32')


# data we will search through
wb = read_fvecs('./sift/sift_base.fvecs')  # 1M samples
# also get some query vectors to search with
xq = read_fvecs('./sift/sift_query.fvecs')
# take just one query (there are many in sift_learn.fvecs)
xq = xq[0].reshape(1, xq.shape[1])
```

```{code-cell} ipython3
device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
wb = torch.tensor(wb).to(device)
xq = torch.tensor(xq).to(device)
```

```{code-cell} ipython3
wb.shape
```

```{code-cell} ipython3
D = xb.shape[1]
m = 8
assert D % m == 0
nbits = 8  # number of bits per subquantizer, k* = 2**nbits
```

#TODO 
https://www.pinecone.io/learn/product-quantization/

```{code-cell} ipython3

```

```{code-cell} ipython3

```
