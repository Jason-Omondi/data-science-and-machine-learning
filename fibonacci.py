# Function for nth Fibonacci number: a series of numbers in which each number is the sum of the two preceding numbers
import math
#function to check for perfect squares since fibonnaci series consists of numbers with perfect squares
def sq(num):
    s_root = int(math.sqrt(num))
    return s_root*s_root == num

num =  int(input('Enter a number: '))
result1 = 5*(num*num)+4
result2 = 5*(num*num)-4

if(sq(result1) or sq(result2)):
    print(num, 'is a fibonacci number.')
else:
    print(num, 'is not a fibonacci number.')