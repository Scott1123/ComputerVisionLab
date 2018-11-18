import theano
import numpy as np
import theano
import theano.tensor as T

Feat_Score = theano.shared(np.random.randint(0, 10, (10)))
print(Feat_Score.eval())
z1 = T.ones_like(Feat_Score)
print(z1.eval())
z2 = T.concatenate([z1, Feat_Score])
print(z2.eval())
z3 = T.concatenate([Feat_Score, z1])
print(z3.eval())
z_22 = z2[9:]
print(z_22.eval())
z_44 = z3[:11]
print(z_44.eval())
z = z_22 - z_44
z = z[1:10]
print(z.eval())

print(T.sqr(z).eval())
z = T.sum(T.sqr(z))
print(z.eval())
# sub_l2 = T.concatenate([sub_l2, T.stack(z)])
