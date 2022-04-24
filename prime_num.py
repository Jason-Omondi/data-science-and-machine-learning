print("A prime number is a natural number greater than 1 that has no positive divisors other than 1 and itself.\n")

#using the math module to access math functions 
import math 
def prime_2(a):
    try: #wrong user input handling
        type(a) == int
    except Exception as e:
        print('Please enter an integer:' ,e)
    if a == 1:
        return False # 1 is not prime
    max = math.floor(math.sqrt(a))
    for i in range(2,1 + max):
        if a % i == 0:
            return True # is Prime

num = input('Enter a number: ')
#testing our function
for a in range(int(num)+ 1):
    print(a, prime_2(a))

'''#time
import time
t0 = time.time()
for a in range(1,100000):
    prime_2(a)
t1 = time.time()
print('req:', t1-t0)'''
