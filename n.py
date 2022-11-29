import numpy as np
result_df = [1.07900000e+03, 9.45930610e-01, 3.20000000e+01, 7.93128023e-01, 8.02367940e-01, 7.62517102e-01]
print(str(list(map(lambda x : round(x, 3), list(result_df)))).strip('[').strip(']').replace(',', ' '))