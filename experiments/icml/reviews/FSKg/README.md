Comments for reviewer FSKg
--------------------------

The file `perfprofile_dct.pdf` presents a similar experiment to that in Figure 3 of our paper but where $\mathbf{A}$ is generated as a DCT matrix.
More precisely, the matrix $\mathbf{A}$ is firstly generated with entries set as

$$A_{ij} = 2\cos(\tfrac{i\pi(2j+1)}{2\max(m,n)})$$

for all $j \in \{0,\dots,m-1\}$ and $i \in \{0,\dots,n-1\}$. Rows are then randomly shuffled.
In practice, we use `scipy` and $A$ is generated as follows.

```python
import numpy as np
from scipy.fftpack import dct

m = ...
n = ...
A = dct(np.eye(np.maximum(m, n)))
A = A[np.random.permutation(m), :]
A = A[:, :n]
```

The vectors $\mathbf{x}^{\ddagger}$ and $\mathbf{y}$ are generated as explained in Section 4.1 of the paper with the parameters $k=5$, $m=500$,$n=1000$ and $\tau=10$.