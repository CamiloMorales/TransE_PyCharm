from numpy import array
from scipy.sparse import coo_matrix

row  = array([0,0,1,3,1,0,0])
col  = array([0,2,1,3,1,0,0])
data = array([1,1,1,1,1,1,1])

A = coo_matrix( (data,(row,col)), shape=(4,4))
print A
print "######"
print A.todense()

B = coo_matrix( (data,(row,col)), shape=(4,4)).tocsr()
print B
print "######"
print B.todense()