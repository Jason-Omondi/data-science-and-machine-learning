print("A prime number is a natural number greater than 1 that has no positive divisors other than 1 and itself.\n")

# Program to check if a number is prime or not
num = int(input('Enter a number: '))# To take input from the user

# prime numbers function
def prime(num):
    if num > 1: #prime numbers are greater than 1
        # check for factors
        for i in range(2, num):
            if (num % i) == 0:
                # if factor is found, set flag to True
                return True
                # break out of loop
                break
    return False

# check what the function returns: True or False by inserting an if conditional statement
if prime(num):
    print(num, "is not a prime number")
else:
    print(num, "is a prime number")

'''#time
import time
t0 = time.time()
for a in range(1,100000):
    prime_2(a)
t1 = time.time()
print('req:', t1-t0)'''
