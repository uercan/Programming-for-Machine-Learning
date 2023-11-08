# This is a sample Python script.

# Press Mayús+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import matplotlib.pyplot as plt
import math
import pandas as pd

grades = [8,6,1,7,8,9,8,7,10,7,6,9,7]

# task 1 min, max, spread, mean
def min(array):
    minimun = 99
    for number in array:
        if number < minimun:
            minimun = number
    return minimun

def max(array):
    maximum = 0
    for number in array:
        if number > maximum:
            maximum = number
    return maximum

def mean (array):
    sum=0
    for number in array:
        sum = sum +number
    return (sum / len(array))

def variance (array):
    sum = 0
    for number in array:
      sum = sum + (number - mean (array))**2
    return (sum/len(array))

print ('the min is',min(grades),'\nthe max is', max(grades), 'the average value is', mean(grades), 'the variance of the grades is ',\
       variance(grades))


"""
-min function returns the minimum value of an array

-max function returns the maximum value of an array

-mean function returns the mean value of a set of numbers, so the average number calculated by a normal average 

-variance function returns the variance of a set of numbers, it basically tells us how spread is the data that we are working with
the more spread the data is, the larger this value will be
"""

#task 2 (WE HAVE ALREADY CALCULATED THE VARIANCE, IN THIS ONE WE WILL JUST CALCULATE STD)

def std (array):
    return (variance(array)**0.5)

print ('\nThe standar deviation of the dataset is ',std(grades))

"""
Standard deviation tells you how spread out the data is. It is a measure of how far each observed value is from the mean.
In any distribution, about 95% of values will be within 2 standard deviations of the mean
"""

#task 3 MEDIAN AND MEDIAN ABSOLUTE DEVIATION

def order (array):
    aux = []
    ordered = []
    for number in array:
        aux.append(number)
    while array:
        min = array [0]
        for number in array:
            if number < min:
                min = number
        ordered.append(min)
        array.remove(min)

    for number in aux:
        array.append(number)
    return ordered

def median (array):
    ordered = order(array)
    median = ordered[math.floor(len(ordered)/2)]
    return median

def variancemedian (array):
    print('\n\nmedian array function: arraylen is ', len(array))
    sum = 0
    for number in array:
      sum = sum + (number - median (array))**2
    return (sum/len(array))

def stdmedian (array):
    return (variancemedian(array)**0.5)

print('\nthe median value of the grades is: ',median(grades),'\n\nthe median standar deviation is: ', stdmedian(grades))
print ('\n\nThe ordered grades are: ',order(grades))


#TASK 4: plotting our results


plt.hist(grades)
plt.title("histogram")
plt.show()

"""
The most frequent value is 7, which is also the median as we saw before.

There is an outlier at 1, but as they are grades it might be just someone didn't prepare the course well enough
"""
#TASK 5
housing_data=pd.read_csv("housing.csv")
print('\n\n\nThe number of districs in this exercise is:',len(housing_data))

print('\n\nThe average house value in this exercise is: ',mean(housing_data.median_house_value))

#ammount of households

plt.hist(housing_data.households)
plt.title("ammount of households")
plt.show()

#median income

plt.hist(housing_data.median_income)
plt.title("median income")
plt.show()

#housing median age

plt.hist(housing_data.housing_median_age)
plt.title("housing median age")
plt.show()

#median house value

plt.hist(housing_data.median_house_value)
plt.title("median house value")
plt.show()


#D -- Most of the people live in houses which (if the units are as years as we interpret them) are around
# 30 to 35 years old, not so many people live in houses that are new, and that's the minimum column that we
# can see in the histograph


#E -- We think that there's not actually a problem with the data, if anything, it could be that for swedish
#people it might sound weird living in houses with a 100000 value, as they might think that it's in swedish crowns
#but in reality, we are quite sure that those values are in euros, and most of the people live in the range of
#100000 to 200000 euros houses.


#F repeat b and c depending on ocean proximity

housing_nearbay=housing_data[housing_data['ocean_proximity']=="NEAR BAY"]
housing_ocean=housing_data[housing_data['ocean_proximity']=="<1H OCEAN"]
housing_inland=housing_data[housing_data['ocean_proximity']=="INLAND"]

print('\n\nThe average house value for the houses near bay is: ',mean(housing_nearbay.median_house_value))
print('\n\nThe average house value for the houses by the ocean: ',mean(housing_ocean.median_house_value))
print('\n\nThe average house value for the houses inland: ',mean(housing_inland.median_house_value))


#REPEATING TASK C FOR EVERY PROXIMITY

"""NEAR BAY"""


#ammount of households

plt.hist(housing_nearbay.households)
plt.title("ammount of households NEAR BAY")
plt.show()

#median income

plt.hist(housing_nearbay.median_income)
plt.title("median income NEAR BAY")
plt.show()

#housing median age

plt.hist(housing_nearbay.housing_median_age)
plt.title("housing median age NEAR BAY")
plt.show()

#median house value

plt.hist(housing_nearbay.median_house_value)
plt.title("median house value NEAR BAY")
plt.show()



"""OCEAN"""
"""--------------------------------------------------------------"""
#ammount of households

plt.hist(housing_ocean.households)
plt.title("ammount of households OCEAN")
plt.show()

#median income

plt.hist(housing_ocean.median_income)
plt.title("median income OCEAN")
plt.show()

#housing median age

plt.hist(housing_ocean.housing_median_age)
plt.title("housing median age OCEAN")
plt.show()

#median house value

plt.hist(housing_ocean.median_house_value)
plt.title("median house value OCEAN")
plt.show()




"""INLAND"""
"""--------------------------------------------------------------"""
#ammount of households

plt.hist(housing_inland.households)
plt.title("ammount of households INLAND")
plt.show()

#median income

plt.hist(housing_inland.median_income)
plt.title("median income INLAND")
plt.show()

#housing median age

plt.hist(housing_inland.housing_median_age)
plt.title("housing median age INLAND")
plt.show()

#median house value

plt.hist(housing_inland.median_house_value)
plt.title("median house value INLAND")
plt.show()
