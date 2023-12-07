#IMPORTING LIBRARIES

import scipy.special as sp # UNUSED?
import numpy as np # NumPy offers comprehensive mathematical functions, random number generators, linear algebra routines, Fourier transforms, and more
import matplotlib.pyplot as plt # Provides a MATLAB-like way of plotting
import pandas as pd # Data analysis and manipulation tool. Used to read in bathymetry data (M77T file), Directions data (excel file), and organises Vmean into a labelled data structure
# M77T data files are created from raw data collected by marine researchers using InfoBank
from scipy.interpolate import CubicSpline as SP # Used to create a spline about the velocity, shape parameter and proability data
from scipy.optimize import minimize,least_squares,Bounds,basinhopping # Optimizing and route finding
from scipy.special import gamma # Gamma function used for weibull distribution
import matplotlib.cm as cm # UNUSED?
from Boruvka_Mod import Graph # imports class graph from Boruvka_Mod python file
import utm # Used to help convert latitude and longitude in bathymetry data to centred eastings and northings
import scipy.interpolate as spline # Used to smooth the bathymetry data
from matplotlib.colors import ListedColormap # UNUSED?
import math # Provides access to the mathematical functions
from time import perf_counter # Returns the float value of time in seconds. Used to record run time for clustering and optimization algorithms
from minmax_kmeans import minsize_kmeans # imports minsize_kmeans function from minmax_kmeans python file
import pulp #Used in the clustering algorithm to generate MPS and LP files
import scipy.signal as signal 
from scipy.signal import savgol_filter
from scipy.interpolate import make_interp_spline, BSpline
from scipy.spatial import ConvexHull
from GP_functions1 import GP_train, GP_predict

import pygad
import warnings
warnings.filterwarnings('ignore')




from GENETICFUNCTIONS import GeneticFunctions
genetic = GeneticFunctions()

import parameters as p


initialPopulation,xGrid,yGrid = genetic.initialPositions(p.xDimension,p.yDimension,p.numTurbs,p.solPerPopTest)

# Import FunctionToCall, then use FunctionTocall.function

# Submit as different jobs with different number of turbines (x)
# Job script is in shell script, which calls the .py file.


#READING IN BATHYMETRY DATA
data = pd.read_csv("nsea86.m77t", sep='\t') # Reads in all data from file including depth data, lattitude, longitude etc.
mindepth = min(data['CORR_DEPTH']) # Retrieves minimum depth value from data file 'nsea86.m77t'    
print('Shallowest depth:', mindepth, 'm') #Prints out the result for the smallest depth value

maxnodespertree = p.maximumturbinespertree

# Shoreline implementation
initialPopulation,xGrid,yGrid = genetic.initialPositions(p.xDimension,p.yDimension,p.numTurbs,p.solPerPopTest)
positionlist = initialPopulation[0]
positionlist = np.reshape(positionlist,(-1,2))


XShoreLine = np.array([np.linspace(min(positionlist[:,0])-2500,max(positionlist[:,0])+2500,1000)]) #Creates x coordinates of shoreline bewtween min and max turbine locations +/- 1000m # 1000 data points
YShoreLine = np.array([np.linspace(-p.DistanceToShore,-p.DistanceToShore,1000)]) #Creates y coordinates of shoreline at -(distance of the wind farm from the shore) # 1000 data points
TXShoreLine = np.transpose(XShoreLine) #transpose of x coordinates
TYShoreLine = np.transpose(YShoreLine) #transpose of y coordinates
TShoreLine = np.concatenate((TXShoreLine,TYShoreLine),axis=1) # Shoreline (x,y)

OnshoreSubstation,ExportDistance = genetic.Substation_To_Shore_Connection(positionlist)

# Pushes wind farm further out if the smallest distance between the shore and center of wind farm is too small
# Only used for the pre-optimised layout
#In terms of keeping it this way there are bounds on (minimize(targetfunction, initial guess, bounds,.....etc.) the optimization function

CloseTurbY = min(positionlist[:,1]) # y coordinate of closest turbine
CloseTurbYindex = np.argmin(positionlist[:,1]) # index from closest turbine on y axis
CloseTurbX = positionlist[CloseTurbYindex,0] # x coordinate of closest turbine on y axis
DistanceToMinTurb = min(np.sqrt((CloseTurbX-TXShoreLine[:,0])**2+(CloseTurbY-TYShoreLine[:,0])**2)) #min distance from shore to closest turbine 
s = 0

if DistanceToMinTurb < p.TooCloseShore: # if closest turbine is too close to the shore
    for s in range(0,p.numTurbs+1): # loops for number of turbines plus the substation
        positionlist[s,1] = positionlist[s,1] + (p.TooCloseShore-DistanceToMinTurb)
        # Shifts entire wind farm up by the difference to achieve the minimum distance required
    genetic.Substation_To_Shore_Connection(positionlist) #Calls function to redo substation to shore connection with new values

LandCost = genetic.LandAreaCost(positionlist,ExportDistance)

t1_start = perf_counter() # Starts timer
indiceslist,u_labels,label = genetic.clustering_algorithm(positionlist,p.numTurbs,maxnodespertree) #Executes clustering algorithm
t1_stop = perf_counter() # Ends timer
print('Function run time:', t1_stop-t1_start) # Prints algorithm run time

MSTweight=genetic.Minimum_Spanning_Tree(genetic.geom_analysis(positionlist, indiceslist)[0],indiceslist)
links = [None] * math.ceil(p.numTurbs/maxnodespertree)
MSTWeightSum = 0 #MST weight is simply the total length of cabling used

for a in range(0,math.ceil(p.numTurbs/maxnodespertree)):
    MSTWeightSum += MSTweight[a][0]
    print('Each tree weight', 'tree',a+1 ,MSTweight[a][0])
    links[a] = MSTweight[a][1]

#CONVERTS LATITUDE AND LONGITUDE IN BATHYMETRY DATA TO CENTERED EASTINGS AND NORTHINGS

data['easting'] = data.apply(lambda row: utm.from_latlon(row['LAT'], row['LON'])[0], axis=1) #Converts lon/lat to easting and adds this onto the 'data' file
data['northing'] = data.apply(lambda row: utm.from_latlon(row['LAT'], row['LON'])[1], axis=1) #Converts lon/lat to northing and adds this onto the 'data' file

zero_east = min(data['easting']) + (max(data['easting']) - min(data['easting']))/2 +30000 # Determines centre of bathymrtry data in easting direction (+30,000 is simply to get a more interesting result from the current bathymetry data)
zero_north = min(data['northing']) + (max(data['northing']) - min(data['northing']))/2 # Determines centre of bathymetry data in northing direction   

data['centered_easting'] = data['easting'] - zero_east # Centres the data about (0,0) and adds this onto the 'data' file
data['centered_northing'] = data['northing'] - zero_north # Centres the data about (0,0) and adds this onto the 'data' file


maxdepth = max(data['CORR_DEPTH'])
levels = np.linspace(mindepth,maxdepth,24) # Creates 24 even spaces or levels between the min and max depth

smooth = spline.SmoothBivariateSpline(data['centered_easting'], data['centered_northing'], data['CORR_DEPTH'], s = 25000)

#SETTING MODEL PARAMETERS

v0=8.0  #Incoming wind speed
nwind = 33
wind=np.arange(254,287)

#b=np.array((1,5,1000))
#b=np.array(( 1.40002561e-01,   8.51478121e+00,   2.62606729e+03))

# 3 model parameters below:

# 0th: how much power the turbine removes at the centre of the peak of the power distribution.
# 1st: how wide the angle of effect is. UNITS: degrees
# 2nd: up to how far back the effect takes place (approx. 2.6km) UNITS: [m]

model=np.array((1.39998719e-01, 8.51483871e+00, 2.62613638e+03))

ws=2.0 #weibull scale factor
wei_gamma=gamma(1.+1./ws)

Pr = 2*10**6 #Rated Power for Horns Rev 1 Turbines. Max power output [Watts]
Vc = 4.0 #Cut-in Velocity. Starts producing energy at wind speed of 4m/s [m/s]
Vr = 15.0 #Rated Velocity. Starts producing max energy at 15m/s
Vf = 25.0 #Cut-off Velocity. Turbines cut out at wind speeds of 25m/s to prevent damage to the turbines.
k = 2.0 #Weibull shape parameter
(Pr,Vc,Vr,k)

#interpolation parameters
dvel=1.5 #[m/s]
dang=1. #[degrees]

#DETERMINING WIND DIRECTION AND VELOCITY AT HORNS REV 1

v=np.loadtxt('hornsrev_data_all.txt') # assigns data to variable v.
vxref=v[:,3] # (INDEXING STARTS FROM 0). Third column of data is velocity of wind in x direction.
vyref=v[:,4] # (INDEXING STARTS FROM 0). Fourth column of data is velocity of wind in y direction.
angles=v[:,0] # Zeroth column of data is wind angle.
vmean=np.sqrt(np.square(vxref)+np.square(vyref)) # Uses pythagoras to find the wind magnitude + direction for each location.

vmean=pd.DataFrame(vmean) # organises vmean into labelled data structure
vref=pd.DataFrame()

vmean['angle']=angles # add another column to vmean (angle)
vref=vref.append(vmean) # add empty pandas data frame

vref=vref.groupby('angle')
vref.groups
vref.describe()
vref.get_group(260).iat[50,0] # data manipulation to group by angle etc.

#Arrays for pairwise distances and angles
#Angle 0: x direction. Angle pi/2: y directıon

# calculates distance and angle between each pair of turbines:

distance=np.zeros((p.numTurbs,p.numTurbs)) # 2x2 matrix of distances between turbines i and j where distance i-i and distance j-j = 0
# as they're distances to themselves: i-i i-j
                                 #    j-i j-j

angle=np.zeros((p.numTurbs,p.numTurbs)) # same as above but for angles between turbines.

for i in range(0,p.numTurbs):
   
    # 80x80 matrices as there's 80 turbines at Horns Rev 1.
    # squares y distance and x distance then sqrt to find overall distance between 2 turbines.
    distance[i,:]=np.sqrt(np.square(positionlist[i+1,0]-positionlist[1:,0])+np.square(positionlist[i+1,1]-positionlist[1:,1]))
    # same as above using arctan2 whilst giving correct quadrant (between 2 turbines).
    angle[i,:]=np.arctan2(positionlist[1:,1]-positionlist[i+1,1],positionlist[1:,0]-positionlist[i+1,0])
# Rotate angles so that north=0 and convert to degrees (and clockwise instead of anticlockwise)
angle=-np.rad2deg(angle)+270

# Rotating angles to wind direction
windangle=1 # degrees
rotangles=np.mod(angle-windangle+180,360)-180

# IMPORTING WIND DIRECTION DATA

values = genetic.att(p.r,np.mod(p.theta+np.pi,np.pi*2)-np.pi,model) # gets angles from -pi to pi, not from 0 to 2pi.

# TRAIN MODEL and assign to predicting class
name = "all_dataset.csv"
training_class = GP_train() # create training class
training_model = training_class.train_model() # trains model on "all_dataset.csv"

# CREATE PREDICTING CLASS from which attenuation predictions are made
predict_class = GP_predict(training_model) # create predicting class

velocities=np.arange(Vc,Vf,dvel) # 1D array ranging from Vc to Vf in dvel intervals. 4 to 25 in steps of 1.5
angles=np.arange(0,360,dang) # 0 to 360 with intervals of dang.
wsp=genetic.windspeedprobability(angles,velocities,dang,dvel) # tells probability of getting wind from a certain direction with a certain velocity.
# sum of all numbers would add up to 1.
#plt.plot(x,y)

# CALCULATES THE POWER CURVE OF A TURBINE

# Plots the power curve of a turbine.
# No power output from 0-4m/s, then cubic rise up to 15m/s and stays
# constant until cut-off velocity of 25m/s.

#cubic Based Power output
a=Pr/(Vr**3-Vc**3)
b=Vc**3/(Vr**3-Vc**3)




x=np.arange(0,25,0.1)
POvec=np.vectorize(genetic.q) # vectorises power output function q (from cell above).
y=POvec(x)

# PLOTS TURBINE POWER CURVE
# Nothing needed??

wsr=genetic.windspeedreduction(positionlist[1:,:],angles,predict_class)
powout=genetic.power(wsr,velocities)

# EXPECTED TURBINE OUTPUT (Watts) for the given wind distribution and positions.
# Can be seen that top left (NW) produces most energy for Horns Rev 1 Wind Farm.

# EXPECTED POWER FROM WIND FARM CALCULATIONS
# multiply the power that you'd get for each wind angle and speed condition by the probability of this occuring, for all angles and speed
# values and sum to find the expected power.
# Sum up this value for all turbines to find total wind farm expected power output.
output=np.zeros((p.numTurbs)) 
output=np.tensordot(powout,wsp,2)
print(output,np.shape(output),np.sum(output))
# last output is total power output in Watts.

# TURBINE OUTPUT (Watts) WITH NO INTERFERENCE

#Reference output:
# Same calculations as cell above except each turbine sees the full amount of wind every time
# (i.e. no wind reduction from other turbines).
refpowout=genetic.power(np.ones((p.numTurbs,np.size(angles))),velocities)
refoutput=np.zeros((p.numTurbs))
refoutput=np.tensordot(refpowout,wsp,axes=2)
print(refoutput,np.shape(refoutput),np.sum(refoutput))

# last output is total power output in Watts.

distance,angle,distance_global, angle_global = genetic.geom_analysis(positionlist,indiceslist)
wsr=genetic.windspeedreduction(positionlist[1:,:],angles,predict_class)
powout=genetic.power(wsr,velocities) # total output power
output=np.tensordot(powout,wsp,axes=2)

df = pd.DataFrame(data={'No. Turbines':[],
                        'Generation':[],
                        'Turbine Coords':[]})

fitness_function = genetic.fitness_func

# +1 to account for the substation position
# *2 for x and y coordinate for each
num_genes = (p.numTurbs + 1) * 2
init_range_low = 0
init_range_high = 100

# Determines the type of genetic algorithm used
parent_selection_type = "sss"
crossover_type = "single_point"
mutation_type = "random"

# Store the time that execution began
t1_start = perf_counter() 





wsp=genetic.windspeedprobability(angles,velocities,dang,dvel) # tells probability of getting wind from a certain direction with a certain velocity.







# This for loop allows for multiple tests to be executed consecutively
for test in range(0,p.numTests):
    print(f"Simulation {test+1} of {p.numTests} completion:")
    
    # Access the algorithm characteristics from the relevant lists
    numGenerationsTest = p.numGenerations[test]
    solPerPopTest = int(p.solPerPop[test])
    mutationPercentGenesTest = p.mutation_percent_genes[test]
    
    # Define the operation of the instance of the genetic algorithm
    ga_instance = pygad.GA(num_generations=numGenerationsTest,
                       num_parents_mating=p.numParentsMating,
                       fitness_func=fitness_function,
                       sol_per_pop=solPerPopTest,
                       init_range_low=init_range_low,
                       init_range_high=init_range_high,
                       parent_selection_type=parent_selection_type,
                       keep_parents=p.keepParents,
                       crossover_type=crossover_type,
                       mutation_type=mutation_type,
                       mutation_percent_genes=mutationPercentGenesTest,
                       gene_space=p.xRange,
                       initial_population=initialPopulation,
                       save_best_solutions=False) 
    
    # Run the genetic algorithm
    ga_instance.run()
    # Save the best solution
    solution, solution_fitness, solution_idx = ga_instance.best_solution()
    # Reshape the solution genes to represent coordinates
    solutionCoords = np.reshape(solution, (-1, 2))
    solutionCoords = np.array(solutionCoords)

# Mark the time at which processing is complete
t1_stop = perf_counter()
# Calculate run time
totaltime = t1_stop - t1_start # total run time in secondspositionlist = solutionCoords
print(totaltime)

# Output the df to a file such as Test-10-1.csv
df.to_csv('Test-'+str(p.numTurbs)+'-'+str(windangle)+'.csv',index=False)

# DEBUGGING TEST FOR TURBINE CLASHES
for turb in range(0,p.numTurbs+1):
    for checked in range(0,turb):
        if positionlist[turb][0] == positionlist[checked][0] and positionlist[turb][1] == positionlist[checked][1]:
            #If a turbine is too close, set the fitness to be very bad
            print("TURBINE ON TOP --- FAIL")
            
print(solution_fitness)