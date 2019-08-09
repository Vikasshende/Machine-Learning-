import statistics

n=int(input())
print(n)

x=list(map(int,input().split()))
print(x)

a=statistics.mean(x)
print(a)

c=statistics.median(x)
print(c)

d=statistics.stdev(x)
print(d)

b=statistics.mode(x)
print(b)