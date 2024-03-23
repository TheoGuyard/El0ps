Comments for reviewer FSKg
--------------------------

The file `perfprofile_dct.pdf` presents a similar experiment to that in Figure 3 of our paper but where $\mathbf{A}$ is generated as a DCT matrix.
Instances are generated as in Section 4.1 with the parameters $k=5$, $m=500$,$n=1000$ and $\tau=10$.
The matrix $\mathbf{A}$ is generated as follows.

```python
import numpy as np
from scipy.fftpack import dct

m = ...
n = ...
A = dct(np.eye(np.maximum(m, n)))
A = A[np.random.permutation(m), :]
A = A[:, :n]
```