#!/usr/bin/env python
# coding: utf-8

# In[3]:


import math
from scipy.stats import norm
#defining the variables
S= 80 #Underlying Price
K= 70 #Strike Price
T= 0.5 #Time to Maturity
r= 0.03 #Risk-free Rate
vol= 0.2 #Volatility
d1= (math.log(S/K)+(r+0.5*vol**2)*T)/(vol*math.sqrt(T))
d2= d1-(vol* math.sqrt(T))
#call option price calculation
C= S * norm.cdf(d1)-K*math.exp(-r*T) * norm.cdf(d2)
# put option calculation
P= K * math.exp(-r*T) * norm.cdf(-d2)-S*norm.cdf(-d1)
print('The Value of d1 is:',round(d1,4))
print('The value of d2 is:', round(d2,4))
print('The price of the call option is $',round(C,2))
print('The price of the put option is:$',round(P,2))


# In[ ]:




