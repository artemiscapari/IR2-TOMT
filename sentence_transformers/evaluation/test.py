from ..util import cos_sim, dot_score
from torch import tensor
from torch import rand

q = rand((100, 768))
d = rand((100, 768))

print(cos_sim(q,d))
