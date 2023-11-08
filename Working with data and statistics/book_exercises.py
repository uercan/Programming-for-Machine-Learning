###EXERCISE 3.2

def do_twice(f, a):
    f(a)
    f(a)
def do_four(f,a):
    do_twice(f,a)
    do_twice(f,a)

def print_twice(bruce):
    print(bruce)
    print(bruce)


do_twice(print_twice, "spam")
do_four(print_twice,"test")

"""-----------------------------"""

#EXERCISE 5.3
def is_triangle(a,b,c):
    if a+b>=c and a+c>=b and b+c>=a and b!=0 and a!=0 and c!=0:
        print("yes")
    else:
        print("no")

is_triangle(0,5,5)

lenght_a = input("enter a length for the side a: ")
lenght_b = input("enter a length for the side b: ")
lenght_c = input("enter a length for the side c: ")
print("\n\n the lengths of the triangle are: a: ",lenght_a, " b:", lenght_b, " c: ",lenght_c)

is_triangle(lenght_a,lenght_b,lenght_c)


"""-----------"""
#EXERCISE 8.4

def any_lowercase1(s):
    for c in s:
        if c.islower():
            return True
        else:
            return False
#it returns true if the first character is lowercase and false if the first character is uper case

def any_lowercase2(s):
    for c in s:
        if 'c'.islower():
            return 'True'
        else:
            return 'False'
#in this case, it doesn't really matter what parameter you send to the function as it's checking
#if the character 'c' is lower case or not, as before, it just iterates once as it gets a return in
#the first iteration

def any_lowercase3(s):
    for c in s:
        flag = c.islower()
            return flag

def any_lowercase4(s):
    flag = False
    for c in s:
        flag = flag or c.islower()
    return flag

def any_lowercase5(s):
    for c in s:
        if not c.islower():
            return False
    return True