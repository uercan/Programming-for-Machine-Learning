import random
from math import sqrt

import numpy as np
import matplotlib.pyplot as plt
from Tools.scripts.make_ctype import values
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

###TASK 1
print("\nTASK1")
f1 = open ("pop_year_trim.csv")
f2 = open ("income_data.csv")

workingdata2=[]

for line in f2.readlines():
    val=line.split(",")
    workingdata2.append(val)

list2header=workingdata2[0].copy()
del workingdata2[0]

agewage = {}
for i in range (16, 101):
    for j in workingdata2:
        if j[1].__contains__(str(i)):
            if i in agewage:
                agewage[i]=agewage[i]+float(j[2])/21
            else:
                agewage[i]=float(j[2])/21
xdata=np.array(list(agewage.keys()))
ydata=np.array(list(agewage.values()))
xdata=xdata.reshape(-1,1)

lin_reg = LinearRegression()
lin_reg.fit(xdata, ydata)
print("Estimated model parameters", lin_reg.intercept_, lin_reg.coef_)

X_new = np.array([[35], [80]])
y_predict = lin_reg.predict(X_new)
print("\nPrediction for the new data points", y_predict)

##Mean squared error
mse = 0
for i in range (16, 101):
    mse = (mse + (ydata[i-16]-(lin_reg.predict(np.array([i]).reshape(-1,1))))**2)
mse = mse/85

print("\nThe mean squared error of th emodel is: ", mse)

"""
There are not any differencies between the model that we have calculated previously in the second lab manually and this 
model that scikit calculated for us
"""

##TASK 2
print("\nTASK2")
##TASK 2.1, implement the polynomial regression
poly_features = PolynomialFeatures(degree=3, include_bias=False)
X_poly = poly_features.fit_transform(xdata)
lin_reg2 = LinearRegression()
lin_reg2.fit(X_poly, ydata)
print("\nEstimated model parameters with a 3rd degree polynomial", lin_reg2.intercept_, lin_reg2.coef_)

##TASK 2.2 Grid search, optimize degree for the polynomial fitting
minmse=9999999
error=[]
for i in range (1, 100):
    poly_features = PolynomialFeatures(degree=i, include_bias=False) #We iterate the features of the polynomial for each degree that we want to test
    X_poly = poly_features.fit_transform(xdata) #We fit the x data so it's suitable for the polynomial fitting
    lin_reg2.fit(X_poly, ydata)
    mse2 = 0
    for j in range(16, 101):
        mse2 = (mse2 + (ydata[j - 16] - (#We calculate the mean squared error for each polynomial degree
            lin_reg2.predict(poly_features.fit_transform(np.array([j]).reshape(-1, 1)))))**2)
    mse2 = mse2 / 85
    error.append(mse2)#We append that calculated error to an error array so we can sitck with the minimum of the array

    if mse2 < minmse:
        minmse = mse2
        optimaldegree = i #If the newer error calculated is smaller than the previous smallest error, the smallest error will be updated
                          #and so will be the optimal degree of the polynomial
print ("the optimal degree is: ", optimaldegree)
plt.plot(error)
plt.xlabel('Degree of the polynomial')
plt.ylabel('Average mean squared error')
plt.show()

##TASK 2.3 Graph the results of the polynomial fitting with the previously obtained optimal degree
poly_features = PolynomialFeatures(degree=optimaldegree, include_bias=False)
X_poly = poly_features.fit_transform(xdata)
lin_reg2.fit(X_poly, ydata)
X_new=np.linspace(16, 101, 85).reshape(85, 1)
X_new_poly = poly_features.transform(X_new)
y_predict = lin_reg2.predict(X_new_poly)
#Here comes the plot
plt.plot(xdata, ydata, "b.")#this is the real data that we have
plt.plot(X_new, y_predict, "r-", linewidth=2, label="Predictions")#this is the prediction
plt.show()


##TASK 3
print("\nTASK3")
#3.1 Just opening the new data
workingdata3 = []
f3 = open ("inc_vs_rent.csv")
for line in f3.readlines():
    val=line.split(",")
    workingdata3.append(val)
print ("\nThe new dataset looks like this: ",workingdata3)

##just by looking at the table we can see the price of the SQM for the annual rent in SEK, so basically how much it costs
##to rent a square meter for a year depending on where you are, as well as the average yearly income in KSEK (thousand of crowns)

#3.2 Create a scatter plot of my points
list3header=workingdata3[0].copy()
del workingdata3[0]
annualrentsqm=[]
avguearlyincome=[]
for i in range(len(workingdata3)):
    annualrentsqm.append(float(workingdata3[i][3]))
    avguearlyincome.append(float(workingdata3[i][4]))
#So far I've just prepared the data to be plotted in two same size arrays
plt.plot(annualrentsqm, avguearlyincome, "b.")#this is the real data that we have
plt.xlabel('annual rent per SQM')
plt.ylabel('average yearly income')
plt.show()

#3.3 K means implementation
"""
The way that the Kmeans class that I have implemented receives the data argument, is in a dictionary, so let's give format
to the data, as well as the number of clusters we want to have
"""
datalist=[]
for i in range(len(annualrentsqm)):
    auxlist=[annualrentsqm[i], avguearlyincome[i]]
    datalist.append(auxlist)
print(datalist)

datadict = {}
for i in range(len(datalist)):
    datadict["x"+str(i)] = datalist[i]


class Kmeans:
    def __init__(self, k, data):##We initialize the object here, we have to call it with the number of clusters and with the
                                ##input data
        self.datapoints = data
        self.clusters = {}
        self.__numberofclusters = k
        self.datainlist = list(self.datapoints.values())
        self.xdata = []
        self.ydata = []
        for i in range(len(self.datainlist)):
            self.xdata.append(self.datainlist[i][0])
            self.ydata.append(self.datainlist[i][1])
        self.__upperboundx = max(self.xdata)  # we create the boundaries in within our initial centroids must be for the randomization algorithm
        self.__lowerboundx = min(self.xdata)  #not to exceed them
        self.__upperboundy = max(self.ydata)
        self.__lowerboundy = min(self.ydata)
        self.__initcentroids = self.initialcentroids()
    def initialcentroids(self):
        centroids = {}
        for j in range(self.__numberofclusters):
            centroids[j] = [random.uniform(self.__lowerboundx, self.__upperboundx),random.uniform(self.__lowerboundy, self.__upperboundy)]
        return centroids
    def getdistance(self, point1, point2): ##euclidean distance
        distance = sqrt((point1[0]-point2[0])**2+(point1[1]-point2[1])**2)
        return distance
    def recalculatecentroids(self):##
        newcentroids = {} #so we can have the new centroids in the same format as the previous ones for being able to compare them to tell
                          #if the process has finished, we'll put them again in a dictionary
        currentclusterlist =[]
        for n in range(self.__numberofclusters):
            if n in self.clusters:
                currentclusterlist = (list(self.clusters[n])).copy() #We add all the points of the cluster we are iterating to this list
                                                                     #so we just have to average them to get the centroids
                auxcentroidsx = 0
                auxcentroidsy = 0
                for elements in range(len(currentclusterlist)):
                    auxcentroidsx = auxcentroidsx + currentclusterlist[elements][0]/len(currentclusterlist)
                    auxcentroidsy = auxcentroidsy + currentclusterlist[elements][1]/len(currentclusterlist)
                newcentroids[n] = [auxcentroidsx,auxcentroidsy]
            else:
                newcentroids[n] = self.__initcentroids[n] #If there's not any point in a cluster, we remain with the original centroid
            del currentclusterlist[:]
        return newcentroids
    def iterate(self):
        ##calculate all the distances to the centroids from all the points
        centroidsaux=list(self.__initcentroids.values()) #it's easier for calculations in iterations to work with lists rather than dictionaries
        for h in range(len(self.datainlist)):
            listaux = [] #In this auxiliar list we get all the distances from one point to all the different centroids
            for n in range(self.__numberofclusters):
                listaux.append(self.getdistance(self.datainlist[h], centroidsaux[n]))
            for c in range(len(listaux)):
                if listaux[c] == min(listaux): #The point will belong to the cluster whose centroid is closer to the point itself
                    if c in self.clusters: #if the cluster already exists
                        self.clusters[c].append(self.datapoints["x" + str(h)])
                    else:#if the cluster doesn't exist we have to initialize it first
                        self.clusters[c] = []
                        self.clusters[c].append(self.datapoints["x"+str(h)])
                        #We use append instead of an assignment with an "=" because that way we can have a list of lists
                        #inside the dictionary key, otherwise all the values will be one after the other and there's not gonna
                        #be a way to control separately x and y values.

                        #There is a chance that even if we specified K clusters we'll have less as maybe one of the centroids is not
                        #closer to any point, so even if there is going to be a centroid for that cluster, there won't be any points in that cluster
            del listaux[:]
        nextcentroids = self.recalculatecentroids()
        if self.__initcentroids == nextcentroids: ##previous centroids is the same as init centroids,  just didn't wanna create another variable
            return nextcentroids
        for c in range(len(self.clusters)):
            if c in self.clusters:
                del self.clusters[c]
        self.__initcentroids = nextcentroids
        return self.iterate() #it's a recursive function, there might be times in which we'll get an error due to infinite iterations looping here

##TASK4 silhouette score for iterating and checking the optimal value of K
print("\nTASK4")
def insidedistance (finalclusterslistfunc): #We get the average distance here between a point and all the other points from the same cluster a(i)
    WithinClusterDistance = []
    for j in range(len(finalclusterslistfunc)):
        for i in range(len(finalclusterslistfunc[j])):
            aux = 0
            for b in range(len(finalclusterslistfunc[j])):
                 aux = aux + sqrt((finalclusterslistfunc[j][i][0]-finalclusterslistfunc[j][b][0])**2+(finalclusterslistfunc[j][i][1]-finalclusterslistfunc[j][b][1])**2)
            aux = aux/(len(finalclusterslistfunc[j]))
            WithinClusterDistance.append(aux)
    return WithinClusterDistance

def outsidedistance(finalclusterslistfunc, finalcentroidslistfunc): #b(i)
    # We get the average distance here between a point and all the other points from the closest cluster not considering the cluster that it belongs to
    ClosestClusterDistance = []
    for j in range(len(finalclusterslistfunc)):
        for i in range(len(finalclusterslistfunc[j])):
            MinDistCentroid = 999
            for z in range(len(finalclusterslistfunc)): #Here we are gonna make sure we don't compare with the same centroid of the cluster
                                                        #that the point belongs to already
                if z == j:
                    continue
                else:
                    CentDistance = sqrt((finalclusterslistfunc[j][i][0]-finalcentroidslistfunc[z][0])**2+(finalclusterslistfunc[j][i][1]-finalcentroidslistfunc[z][1])**2)
                    if CentDistance < MinDistCentroid:
                        MinDistCentroid = CentDistance
                        closestcluster = z
            aux = 0
            for m in range(len(finalclusterslistfunc[closestcluster])):
                aux = aux + sqrt((finalclusterslistfunc[j][i][0] - finalclusterslistfunc[closestcluster][m][0]) ** 2 + (finalclusterslistfunc[j][i][1] - finalclusterslistfunc[closestcluster][m][1]) ** 2)
            aux = aux / len(finalclusterslistfunc[z])
            ClosestClusterDistance.append(aux)
    return ClosestClusterDistance

def silhouette(finalclusterslistfunc1, finalcentroidslistfunc1): #with the previous functions combined here we calculate the silhouette score s(i)
    distins = insidedistance(finalclusterslistfunc1)
    distout = outsidedistance(finalclusterslistfunc1, finalcentroidslistfunc1)
    silpoints = []
    for i in range(len(distins)):
        silpoints.append((distout[i] - distins[i]) / (max(distout[i], distins[i])))
    #print(silpoints)
    avg = 0
    for j in range(len(silpoints)):
        avg = avg + silpoints[j]/len(silpoints)
    return avg

silhouettescores = []
for k in range (2, 6): #now we iterate on K values within a range which in this case is from 2 to 6 so it'll test 2, 3, 4 and 5
    Ktest = Kmeans(k, datadict)
    finalcentroidstest = Ktest.iterate()
    finalclusterslisttest = list(Ktest.clusters.values())
    finalcentroidslisttest = list(finalcentroidstest.values())
    silhouettescores.append(silhouette(finalclusterslisttest, finalcentroidslisttest))
print ("\nThe array of silhouettescores obtained is: ",silhouettescores)
for i in range(len(silhouettescores)):
    if silhouettescores[i] == max(silhouettescores):
        optimalK = i+2
        print("\nThe optimal value of K is: ",optimalK)
vector = range(2,6)

plt.plot(vector, silhouettescores)
plt.xlabel('different Ks')
plt.ylabel('silhouette scores')
plt.show()

datadictnew = datadict
datadictnew["x21"] = [1010, 320.12] ##After a few executions this appears to be in a proper cluster, closer to similar members
datadictnew["x22"] = [1258, 320.00] ##After a few executions this appears to be in a proper cluster, closer to similar members
datadictnew["x23"] = [908, 292.4] ##Sometimes this one appears to be alone in a separate cluster, other times in the same cluster as the
                                  ##one that's to the right of it, it is due to not adding this clusters to the gridsearch, as it is a bit appart from
                                  ##the rest of the data, it's noticeable than when we use a higher K this point is almost always going to have it's own
                                  ##cluster, or at least in most of the ocassions
Ktry = Kmeans (optimalK, datadictnew)
finalcentroids = Ktry.iterate()
finalclusterslist = list (Ktry.clusters.values())
print("\nThe final cluster points are (if there are not as many as the initial K specified, that's because the initial centroids \
were not good,\n so one or some centroids might not have any points which are closer to those centroids than to \
other centroids)\n",finalclusterslist)
finalcentroidslist = list(finalcentroids.values())
print("\nThe final centroids obtained are: ",finalcentroidslist)
print("\nThe clusters obtained are: ",Ktry.clusters)

xdataclusters=[]
ydataclusters=[]
xdataclustersaux=[]
ydataclustersaux=[]

for i in range(len(finalclusterslist)):
    for p in range(len(finalclusterslist[i])):
        xdataclustersaux.append(finalclusterslist[i][p][0])
        ydataclustersaux.append(finalclusterslist[i][p][1])
    xdataclusters.append(xdataclustersaux.copy())
    ydataclusters.append(ydataclustersaux.copy())
    del xdataclustersaux
    del ydataclustersaux
    xdataclustersaux = []
    ydataclustersaux = []

for u in range(len(xdataclusters)):
    plt.scatter(xdataclusters[u], ydataclusters[u])
plt.xlabel('annual rent per SQM')
plt.ylabel('average yearly income')
plt.show()
