import pandas as pd
import numpy as np
d=pd.read_html('https://github.com/akiwelekar/MLModels/blob/master/aimarks2017.csv')
df=d[0]
mse_marks=np.array(df['mse'])
ese_marks=np.array(df['ese'])
x = pd.Series(mse_marks)

y = pd.Series(ese_marks)

r = x.cov(y) / (x.std() * y.std())

beta1 = ( r* (y.std())) / x.std()

beta0 = y.mean()  - (beta1 * x.mean())

print("slope=%.2f"%(beta1))
print("intercept:%.2f"%(beta0))
