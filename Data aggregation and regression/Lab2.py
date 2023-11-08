#TASK 1 inspect and load the data
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import string



def openfile (filepath):
    f = open(filepath)
    return f

f1 = openfile ("pop_year_trim.csv")
#text=f1.read()
#print (text)

#TASK 2 filtering the data

#first we are gonna convert the data to lists
workingdata=[]
for line in f1.readlines():
    val=line.split(",")
    workingdata.append(val)

listheader=workingdata[0].copy()
del workingdata[0]

print ("this is the workingdata header:\n",listheader)
filteredlist=[]
for i in workingdata: ##outerlist corresponds to rows
    if i[1]=="25 Norrbotten county":
        if i[2]=="post secondary education":
            filteredlist=i.copy()

print ("\nthis is the filtered data: \n",filteredlist)

averagepopulationseceducationnorbotten=(int(filteredlist[3])+int(filteredlist[4])+int(filteredlist[5])+int(filteredlist[6])+int(filteredlist[7]))/5

print ("the mean value of the population with post-secondary education in the Norbotten region in the past 5 years is: "  \
       ,averagepopulationseceducationnorbotten)


def variance (array, mean):
    sum = 0
    for number in array:
        if type(number) is str:
            number=int(number)
        sum = sum + (number - mean)**2
    return (sum/len(array))

variancepopulation=variance(filteredlist[3:8], averagepopulationseceducationnorbotten)

def std (variance):
    return (variance**0.5)

print("\nThe standard deviation of the population in Norbotten who is post secondary education is: ",std(variancepopulation))
#this value is a measure of spread, the larger the variance is, the more spread is the data, which in this case means difference
#in the ammount of people living in Norbotten with post secundary education during the past 5 years.


dict = {}


for i in workingdata:
    if i[1] in dict:
        dict[i[1]] = dict[i[1]] + int(i[7])
    else:
        dict[i[1]]=int(i[7])
#print (dict)


plotnames=list(dict.keys())
plotvalues=list(dict.values())
plt.bar(range(len(dict)), plotvalues, tick_label=plotnames)
plt.title("Population per region in 2020")
plt.show()


##TASK 3 - loading another set of data

f2 = openfile("income_data.csv")
workingdata2=[]
workingdata=[]

for line in f2.readlines():
    val=line.split(",")
    workingdata2.append(val)

list2header=workingdata2[0].copy()
del workingdata2[0]

#print (list2header)
#print (workingdata2)


#In general is better to use the debugger because you can use it mid-execution of the code and it gives you the value of 
#all the variables in real time without having to add any prints, in case that you add a printout, you'll have to delete
#it after you've finished checking your code, which for small cases doesn't really matter but when it's large amount of 
#lines, it might lead to deleting necessary stuff for the correct functioning of the code by error


#TASK 4
agewage = {}
for i in range (16, 101):
    for j in workingdata2:
        if j[1].__contains__(str(i)):
            if i in agewage:
                agewage[i]=agewage[i]+float(j[2])/21
            else:
                agewage[i]=float(j[2])/21
#print (agewage)


xdata = list(agewage.keys())
ydata = list(agewage.values())

A = np.transpose(np.vstack([xdata, np.ones(len(xdata))]))
k, m = np.linalg.lstsq(A, ydata, rcond=None)[0]


multipliedxdata = []
for aux in xdata:
    multipliedxdata.append(aux*k)


plt.plot(xdata, ydata, 'o', label='Original data', markersize=10)
plt.plot(xdata, multipliedxdata+m, 'r', label='Fitted line')
plt.legend()
plt.show()

##predicted values for the points (35, 80)
predicted35= k*xdata[35-16]+m
predicted80= k*xdata[80-16]+m

print ("the predicted values for 35 years old and 80 years old are respectively: ", predicted35, predicted80)
mse = 0


for i in range (16, 101):
    mse = (mse + (ydata[i-16]-(k*xdata[i-16]+m))**2)
mse = mse/85
##this mean square error that we calculated is 9974.389064202915, it's a measure of spread of the data, which means
##how appart the elements are from the average

print ("\n\nThe mean square error of the average yearly income for all the population is", mse, "\n\n")


xdatashorted=xdata[14:].copy()
ydatashorted=ydata[14:].copy()


Ashorted = np.transpose(np.vstack([xdatashorted, np.ones(len(xdatashorted))]))
kshorted, mshorted = np.linalg.lstsq(Ashorted, ydatashorted, rcond=None)[0]


multipliedxdatashorted = []
for aux in xdatashorted:
    multipliedxdatashorted.append(aux*kshorted)

plt.plot(xdatashorted, ydatashorted, 'o', label='Original data shorted', markersize=10)
plt.plot(xdatashorted, multipliedxdatashorted+mshorted, 'r', label='Fitted line shorted')
plt.legend()
plt.show()

predicted35s= kshorted*xdatashorted[35-30]+mshorted
predicted80s= kshorted*xdatashorted[80-30]+mshorted

print ("\n\nthe predicted values for 35 years old and 80 years old for the shorter dataset are respectively: ", predicted35s, predicted80s)

mses = 0
for i in range(30, 101):
    mses = (mses + (ydatashorted[i-30]-(kshorted*xdatashorted[i-30]+mshorted))**2)
mses = mses/71
print ("\nThe mean square error of the average yearly income for the shorter dataset is", mses, "\n\n")

##somehow we get that 35 year old people earn much more when calculating with the shorter dataset, we think it's because
##we are getting rif of the data of people under 30 years old, and they don't have a very high income, which makes the fitted
##line to have a higher slope, which is higher than the real value for 35 years old, whereas in the other case with all the data
##the fitted line is below the real value.

##other thing that we can observe, is that now the error is reduced by a lot, that is maybe as well because the real data
##is hard to fit with just a line, as it is not completely linear, and if we don't have the first elements of the array, up until
##30 years old, the data gets more linear as the income gets more stabilized when it comes to linearity (in other words the real
##data plotted is closer to a line when taking out the first real values, so the linear regression works better)


#######################################################################################################################
#################################################BOOK#EXERCISES########################################################
#######################################################################################################################
##exercise 10.11 book. find reverse elements from a list


def reverselist (array):
    i=1
    newlist=[]
    for element in array:
        aux=array[i:].copy()
        for value in aux:
            if value in newlist:
                continue
            else:
                for j in range(len(value[0])):
                    if value[0][j]==element[0][-j-1]:
                        if j == len(value[0]) -1:
                            newlist.append(value)
                            newlist.append(element)
                    else:
                        break
        i = i+1
    return newlist

testlist=[["cacaxd"],["caca"],["acac"],["amigovol"],["amigo"],["caca"],["ogima"],["esternocleidomastoideo"],["oediotsamodielconretse"],["esternocleidomastoideo"],["esternocleidomastoidea"],["aediotsamodielconretse"]]

reversedelements= reverselist(testlist)
print ("the reversed couples in this array: ", testlist, "\nare: ", reversedelements)


##EXERCISE FROM THE BOOK 11.4 HAS_DUPLICATES FUNCTION
def has_duplicates (array):
    dict={}
    dict["duplicates"] = ''
    for i in array:
        if str(i) in dict:
            dict["duplicates"]=dict["duplicates"] + str(i)
        else:
            dict[str(i)]='exists'
    print ("The duplicate elements are:", dict["duplicates"])


has_duplicates(testlist)


#Example 13.1


#string module should be imported because it contains critical methods like "whitespace" to use in this example.


# fruits.txt contains = Apple, Watermelon, Orange, Pear, Cherry, Strawberry, Nectarine, Grape, Mango,
# Blueberry, Pomegranate, Plum, Banana, Raspberry, Mandarin, Jackfruit, Papaya, Kiwi, Pineapple, Lime, Lemon, Apricot,
# Grapefruit, Melon, Coconut, Avocado, Peach

emptyList=[]

with open("fruits.txt") as f:
  words = f.read().split()
  for elem in words:
    elem = elem.strip(string.punctuation).lower()
    emptyList.append(elem)

print(emptyList)


