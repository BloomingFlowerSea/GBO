from GBO import GBO
from CEC13 import CEC13
import pandas as pd
import numpy as np

fun_num = 1
sum = 0
run_n = 51
function_best = [-1400, -1300, -1200, -1100, -1000, -900, -800, -700, -600, -500, -400, -300, -200, -100, 100, 200, 300,
                 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400]
dim = 30
with pd.ExcelWriter(r'output.xlsx') as writer:
    alg = 'GBO'
    print("alg:", alg)
    the_mean = []
    the_std = []
    the_median = []
    df = pd.DataFrame()
    k = pd.DataFrame()
    for j in range(fun_num - 1, 28):
        print("fun_num:", j + 1)
        all_fit = []
        rosenbrock = lambda x: CEC13(x.reshape((1, dim)), j + 1)
        for i in range(run_n):
            print("RUN:", i)
            gbo = GBO()
            gbo.load_prob(evaluator=rosenbrock)
            best_fit, run_time, best_idv = gbo.run()
            all_fit.append(best_fit - function_best[j])
            print("Best solution:", best_idv)
            print("Best fitness value:", best_fit - function_best[j])
            print("Running time:", run_time, "seconds")
        df['F' + str(j + 1)] = all_fit
        the_mean.append(np.mean(all_fit))
        the_std.append(np.std(all_fit))
        the_median.append(np.median(all_fit))
        print("The fun_num", j + 1, " mean:", np.mean(all_fit))
        print("\n")
    k[str(dim) + 'Dim' + '_mean'] = the_mean
    k[str(dim) + 'Dim' + '_std'] = the_std
    k[str(dim) + 'Dim' + '_median'] = the_median
    sheetk = alg
    sheet = alg + 'all'
    k.to_excel(writer, sheet_name=sheetk, index=False)
    df.to_excel(writer, sheet_name=sheet, index=False)
