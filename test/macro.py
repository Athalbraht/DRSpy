#print(dataset["Delay [ns]"])

def nconv(n, spl="_"):
    return float(n.split(spl)[0])

def t_sum(df, v, w):
    ww = df[w]
    vv = df[v]
    return (ww*vv).sum()/ww.sum()

def asym(df, c1, c2):
    return  (df[c1] - df[c2])/(df[c1] + df[c2])

def rgx(df, x, y):
    xx, yy = df.plot(x, y, regx=True, flive=True)
    return xx, yy
    

print(asym(dataset, "+30_U-CH0", "+30_U-CH1"))
print(t_sum(dataset, "-40_U_t", "Delay [ns]"))

dxr = ".*Delay.*"
dyr = ".*_t"

xd1, yd1 = rgx(dataset, dxr, dyr)


#rx = ".*Channel"
#ry = ".*CH.*"

#xx, yy = ctx.obj.plot(rx, ry, regx=True, flive=True)

#print(xx)

#1dataset.plot()
#plt.show()
