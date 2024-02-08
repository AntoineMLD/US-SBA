from scipy.stats import normaltest

def is_gaussian(df, col):
    stat,p = normaltest(df[col].tolist())
    # print("stat = %.3f, p =%.3f \n" %(stat,p))
    return p > 0.05