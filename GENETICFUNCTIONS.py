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


import parameters as p




#Store all functions within a class
class GeneticFunctions:  
    def __init__(self):
        self.POvec=np.vectorize(self.q)
        # TRAIN MODEL and assign to predicting class
        #name = "all_dataset.csv"
        self.training_class = GP_train() # create training class
        print("BEFORE MACHINE LEARNED")
        self.training_model = self.training_class.train_model() # trains model on "all_dataset.csv"
        print("AFTER MACHINE LEARNED")
        # CREATE PREDICTING CLASS from which attenuation predictions are made

        self.predict_class = GP_predict(self.training_model) # create predicting class

        self.velocities=np.arange(p.Vc,p.Vf,p.dvel) # 1D array ranging from Vc to Vf in dvel intervals. 4 to 25 in steps of 1.5
        self.angles=np.arange(0,360,p.dang) # 0 to 360 with intervals of dang.
        self.wsp=self.windspeedprobability(self.angles,self.velocities,p.dang,p.dvel) # tells probability of getting wind from a certain direction with a certain velocity.
        return

    def a(x,y):
        return x*y
    
    def getGridDimensions(self,numTurbs):
        numRows = int(math.sqrt(numTurbs))
        # Prefer rows = cols or as close as possible
        while numRows > 1:
            if numTurbs % numRows == 0:
                numColumns = numTurbs // numRows
                return numRows, numColumns
            numRows -= 1
        # If prime return 1xN grid
        return 1, numTurbs  
    
    
    def initialPositions(self,xDimension,yDimension,numTurbs,solPerPopTest):
        # Get the initial grid of turbines dimensions
        numRows, numColumns = self.getGridDimensions(numTurbs)
        # Distribute the turbines evenly
        x = np.linspace(0,xDimension,int(numColumns)).astype(int)
        y = np.linspace(0,yDimension,int(numRows)).astype(int)
        # Generate grid from x and y positions
        xGrid, yGrid = np.meshgrid(x, y)
        # Convert 2d lists to 1d
        x = xGrid.flatten()
        y = yGrid.flatten()
        
        # Get pairs of coordinates 
        pairs = np.column_stack((x,y))
        pairs = np.reshape(pairs, (1, -1))
        # Format as required for PyGad
        initialPopulation = pairs.flatten()
        initialPopulation = np.tile(initialPopulation, (solPerPopTest, 1)) 
        
        
        initialPopulation = np.insert(initialPopulation,0,50,axis =1)
        initialPopulation = np.insert(initialPopulation,1,50,axis =1)
        
        return initialPopulation,xGrid,yGrid       
    
    
    def Substation_To_Shore_Connection(self,positionlist):
        OnshoreSubstation = [p.xCoordOnshoreSub,-p.DistanceToShore] # Determines fixed position of onshore substation
        ExportDistance = np.sqrt(((positionlist[0,0]-OnshoreSubstation[0])**2)+((positionlist[0,1]-OnshoreSubstation[1])**2)) # min Hypotenuse for distances between substations
       
        return OnshoreSubstation,ExportDistance


    def LandAreaCost(self,positionlist,ExportDistance):
    
        LandCostperHectare = 17245 # in GBP
        LandCostpermSquared = LandCostperHectare/10000 #  in GBP
        radius = 130 #m
        OnshoreSubLand = np.pi*radius**2 #Land area associated with the onshore substation
        ExportLand = ExportDistance*8 # Length*width 
        
        hull = ConvexHull(positionlist) #computes convex hull using the turbine positions
        LandArea = hull.volume + OnshoreSubLand + ExportLand #Calcualtes convex hull area and adds on the onshore substation land area
    
        LandCost = LandArea*LandCostpermSquared
        
        return LandCost



    def clustering_algorithm(self,positionlist,nturb,maxnodespertree):
        k = math.ceil(p.numTurbs/maxnodespertree) #number of clusters. math.ceil rounds a number upwards to its nearest integer
        [turbines, centres] = minsize_kmeans(positionlist[1:,:] ,k ,min_size = 1, max_size = maxnodespertree) 
        #turbines exists as an array containing the cluster value for each turbine
        #Assigns each turbine to a cluster. excludes substation, executes external python file, 
        #limits for how many turbines are in each cluster between 1 and max nodes per tree 
        label = np.concatenate((np.array([-1]),turbines)) # Adds -1 (substation label) to the other turbine labels
        u_labels = np.unique(label) 
        # finds unique elements of label. Finds the unique cluster labels 
        # each turbine is assigned a cluster/label and this returns a single label for each cluster
        
        a = 0
        indiceslist = np.empty((k,maxnodespertree), dtype=np.int8) # empty array of no.clusters by max nodes per tree
        # indiceslist basically tells us what turbines belong to each cluster
        # each row of indiceslist represents a cluster and each point on that row a indidual turbine within that cluster
        # Each row starts with a 0 which represents the subsation which belongs to all clusters
        # As seen before in the initial layout numbers are assigned starting at 1 which is the turbine (a,0) from the substation
        # The numbering then continues in a cyclic motion traveling anticlockwise up until the total number of turbines
        
        for i in range(k): # loops for number of clusters
            indices = [i for i, x in enumerate(label) if x == a] # enumerate exists as a counter in the loop (starts at x=a=0)
    
            if len(indices) < maxnodespertree:
                noofemptys = maxnodespertree - len(indices)
                for l in range(noofemptys): # For the empty spaces in the matrix replaces these with -100 as fillers
                    indices.append(int(-100))
        
            indiceslist[i,:] = (indices)
            a += 1
        
        arr = np.zeros((k,1),dtype=np.int8)   # zero array of number of clusters by 1 (Substation array)
        indiceslist = np.concatenate((arr,indiceslist),axis=1) # Adds substation zero array to turbine cluster matrix
        return indiceslist,u_labels,label




    def geom_analysis(self,positionlist,indiceslist):
        
        #Arrays for pairwise distances and angles
        #Angle 0: x directıon. Angle pi/2: y directıon
        maxnodespertree = p.maximumturbinespertree
        distance = [None] * math.ceil(p.numTurbs/maxnodespertree) # null array with size based on number of clusters
        angle = [None] * math.ceil(p.numTurbs/maxnodespertree) 
        distance_global = np.zeros((p.numTurbs,p.numTurbs)) #zero array nturb by nturb. This will be for storing the distance between each pair of turbines for cabling
        angle_global = np.zeros((p.numTurbs,p.numTurbs)) # Like above this will be for storing the angle data between each pair of turbines for cabling
        coords = positionlist[1:,:] # Coordinates of each turbine
    
        
        for i in range(0,p.numTurbs): # loops for number of turbines
            distance_global[i,:]=np.sqrt(np.square(coords[i,0]-coords[:,0])+np.square(coords[i,1]-coords[:,1])) # Pythagoras for hypotenuse
            angle_global[i,:]=np.arctan2(coords[:,1]-coords[i,1],coords[:,0]-coords[i,0]) #Pythagoras for angle
            distance_global[i,i]=1e10
            #Rotate angles so that north=0 and convert to degrees
        angle_global=-np.rad2deg(angle_global)+270
        
        for x in range(0,math.ceil(p.numTurbs/maxnodespertree)): # loops for number of clusters
            nturbintree = np.count_nonzero(indiceslist[x,:] > -1) # Everything except the -100 ones which were denoted earlier to show that there is no turbine at that index
            distance[x]=np.zeros((nturbintree,nturbintree))
            angle[x]=np.zeros((nturbintree,nturbintree))
        
            for i in range(0,nturbintree):
                for j in range(0,nturbintree):
                    distance[x][i,j] = np.sqrt(np.square(positionlist[(indiceslist[x,i]),0]-positionlist[(indiceslist[x,j]),0]) + np.square(positionlist[(indiceslist[x,i]),1]-positionlist[(indiceslist[x,j]),1]))
                    angle[x][i,j]=np.arctan2(positionlist[(indiceslist[x,j]),1]-positionlist[(indiceslist[x,i]),1],positionlist[(indiceslist[x,j]),0]-positionlist[(indiceslist[x,i]),0])
                    distance[x][i,i]=1e10
            
            #Rotate angles so that north=0 and convert to degrees
            angle[x]=-np.rad2deg(angle[x])+270
          
        return distance,angle,distance_global,angle_global
    
    
    
    
    
    def Minimum_Spanning_Tree(self,distance,indiceslist):
        maxnodespertree = p.maximumturbinespertree
        MSTweight = [None] * math.ceil(p.numTurbs/maxnodespertree) # null array with sized based on number of clusters
       
        for x in range(0,math.ceil(p.numTurbs/maxnodespertree)): #loops for number of clusters
            nturbintree = np.count_nonzero(indiceslist[x,:] > -1)
            g=Graph(nturbintree) #Graph is part of the external python script titled Boruvka_Mod
            
            for i in range(0,nturbintree):
                for j in range(i+1,nturbintree):
                    g.addEdge(i,j ,(distance[x][i,j]*1000))
               
            MSTweight[x]=g.boruvkaMST()
        return MSTweight
    
    
    
    
    
    def depthvalues(self,positionlist):
        DepthPerTurbine = [None] * (len(positionlist)-1) # none defines a null/no value. Get a null list the size of the np. turbines
        DepthCostPerTurbine = [None] * (len(positionlist)-1) # Get a null list the size of the number of turbines
        DepthCostAllTurbines = np.float64(0) # Assigns initial value to variable
    
        for i in range(0,len(positionlist)-1): # loops for amount of turbines
            DepthPerTurbine[i] = spline.SmoothBivariateSpline.ev(p.smooth, positionlist[i+1,0], positionlist[i+1,1]) # Acquires depth assigned to each turbines position (x,y)
            DepthCostPerTurbine[i] = MinDepthCostPerTurbine = p.CostperTurbine*(0.0002*(float(DepthPerTurbine[i])**2) - 0.0002*(float(DepthPerTurbine[i])) + 0.9459)-p.CostperTurbine #Formula for depth cost
            DepthCostAllTurbines += DepthCostPerTurbine[i] # Sums up all turbines depth costs
            
        mindepth=float(min(DepthPerTurbine))
        maxdepth=float(max(DepthPerTurbine))
        return DepthCostAllTurbines, mindepth, maxdepth # returns respective values to user






    def att(self,dist, ang, model): 
        # 'Model' parameters are defined in 2ND CELL.
    
        # angular part
        angular=np.where(2.*model[1]*np.abs(ang)<np.pi,np.square(np.cos(model[1]*ang)),0.)
        # angular = np.cos(model[1]*ang WHEN 2.*model[1]*np.abs(ang) is less than pi, else angular = 0.
        
        # radial part (distance) (Gaussian Function)
        radial=np.exp(-np.square(dist/model[2])) # decreasing exponential of square, scaled by 2nd parameter
        penalty=np.exp(-np.square(dist/200))
        #penalty = 0
        return 1.0-1*model[0]*angular*radial-2*model[0]*penalty # OUTCOME
    
    
    
    
    def rotate(self,angle,coords):
        angle = np.pi*angle/180.
        rotcoordx = []
        rotcoordy = []
        for coord in coords:
            rotcoordx+=[coord[0]*np.cos(angle)-coord[1]*np.sin(angle)]
            rotcoordy+=[coord[0]*np.sin(angle)+coord[1]*np.cos(angle)]
        rotcoords=[rotcoordx,rotcoordy]
        rotcoords=np.array(rotcoords).T   
        return rotcoords 
    
    
    def q(self,v): # q is power output
            
        if (v<p.Vc): # below cut in velocity power is 0.
            q=0
        elif (v<p.Vr):
            q=p.a*v**3-p.b*p.Pr # cubic power output between cut in and max.
        elif (v<p.Vf):
            q=p.Pr # max power between max and cut off velocity.
        else: 
            q=0 # no power above cut off velocity.
        return q

    
    
    
    def deviation(self,b):
        total_att=np.ones((p.numTurbs,p.nwind))
        deviation=0
        for k in range(0,p.nwind):
            for j in range(0,p.numTurbs):
                for i in range(0,p.numTurbs):
                    if (i!=j):
                        total_att[j,k] = total_att[j,k]*self.att(distance[i,j],np.mod(np.deg2rad(angle[i,j]-wind[k])+np.pi,np.pi*2)-np.pi,b)
                deviation=deviation+np.square(vref.get_group(wind[k]).iat[j,0]-v0*total_att[j,k])
        return deviation
    
    
    # USEFUL OUTPUT USED IN TARGET FN
    # Calculates the power produced by turbines when it sees a certain wind speed at a certain angle. 3D matrix
    def power(self,wsr,v): # wind speed reduction, velocity
        nvel=np.size(v)
        nangle=np.size(wsr,1)
        power_vec=np.zeros((p.numTurbs,nangle,nvel))
        power_vec=self.POvec(np.outer(wsr,v)).reshape(p.numTurbs,nangle,nvel)# np.outer takes every wsr element and individually multiplies it with every element of v.
        return power_vec
    
    
    #GIVES PROBABILITY OF WIND SPEED V AT A GIVEN ANGLE
    def windspeedprobability(self,angles,v,d_angle,d_vel):
        nvel=np.size(v) 
        nangle=np.size(angles)
        wsprob=np.zeros((nangle,nvel))
        for i in range(angles.shape[0]):
            # Get Weibull parameters for angle and evaluate probability
            wsprob[i,:]=(self.wei(v[:],p.wbvel(angles[i]),p.wbshape(angles[i])))*p.windfreq(angles[i])*d_angle*d_vel
        return wsprob
    
    
    def wei(self,x,n,k): # convention to have in this order, scale parameter comes first (x) 
        u=n/gamma(1+1/k) #scaled wind speed
        return (k / u) * (x / u)**(k - 1) * np.exp(-(x / u)**k)
        # k = Weibull shape parameter
        # n = scale parameter
        # x= value we are evaluating
    
    
    def wind_dist(self,v,vm):
        k = 2.0 
        return self.wei(v,vm,k)
    
    
    
    def windspeedreduction(self,positionlist,directions,g_model):
        ndir = int(np.size(directions))
        nturb = int(np.size(positionlist)/2)
        total_att=np.ones((nturb,ndir))
        for i in range(0,ndir):
            angle_to_rotate = 90 + directions[i];
            westerly_pos_list = self.rotate(angle_to_rotate,positionlist) # rotate position list for westerly wind
            att_vector = g_model.predict(westerly_pos_list, nturb) # vector of attenuations (from GP_functions module)
            att_vector = att_vector.reshape(nturb,)
            total_att[:,i] = att_vector
        total_att = total_att/8 ## Get attenuation as a fraction compared to the assumed base wind speed in gaussian model of 8 m/s
        return total_att



    def foundationDepthCost(self,foundation,DepthPerTurbine):
        
        FoundationCost = foundation*(0.0002*(float(DepthPerTurbine)**2) - 0.0002*(float(DepthPerTurbine)) + 0.9459)
        
        return FoundationCost
    



    def foundations(self,positionlist):
        
        Depthx = range(0,math.ceil((max(p.data['CORR_DEPTH']))),1)
        GBCost = [None] * len(Depthx)
        MPCost = [None] * len(Depthx)
        JCost = [None] * len(Depthx)
        TLCost = [None] * len(Depthx)
        Depth = 0

        for i in Depthx:
            GBCost[i] = self.foundationDepthCost(p.GravityBase,Depth)
            MPCost[i] = self.foundationDepthCost(p.Monopile,Depth)
            JCost[i] = self.foundationDepthCost(p.Jacket,Depth)
            TLCost[i] = p.TensionLeg
            Depth += 1
        
        GBCost2 = GBCost[Depthx[0]:Depthx[15]]
        MPCost2 = MPCost[Depthx[15]:Depthx[30]]
        JCost2 = JCost[Depthx[30]:Depthx[60]]
        TLCost2 = TLCost[Depthx[60]:max(Depthx)]
        
        GBMPJTL = GBCost2+MPCost2+JCost2+TLCost2
        smoothed_2dg = savgol_filter(GBMPJTL, window_length = 9, polyorder = 1)
        
        DepthPerTurbine = [None] * (len(positionlist)-1) # none defines a null/no value. Get a null list the size of the np. turbines
        FoundationCost = [None] * (len(positionlist)-1) # Get a null list the size of the number of turbines
        FoundationCostTotal = 0 # Assigns initial value to variable
        
        for i in range(0,len(positionlist)-1): # loops for amount of turbines
            DepthPerTurbine[i] = spline.SmoothBivariateSpline.ev(p.smooth, positionlist[i+1,0], positionlist[i+1,1])
            
            FoundationCost[i] = np.interp(DepthPerTurbine[i],Depthx[0:max(Depthx)],smoothed_2dg)
    
            FoundationCostTotal += FoundationCost[i]
            
        return FoundationCostTotal
    
    
    
    
    
    def Payback(self,TPO,TotalCost,positionlist,numTurbs):
    
        CashFlowYearly = (p.ElectricityCostperkWh*24*365*(TPO/1000))-(p.OperatingandMaintenanceCostsperAnnum*numTurbs) # Calculates the yaerly cash flow for the windfarm
        PayBackTime = TotalCost/CashFlowYearly # calculates payback time in years
        DiscountedPayBackTime = (np.log(1/(1-((TotalCost*p.DiscountRate)/CashFlowYearly))))/np.log(1+p.DiscountRate) #Calculates payback time but takes into account the time value of money
        
        l=0
        NPV = 0
        
        for l in range(p.LifeTimeTurbineOperatingTime):
            l+=1
            NPV = NPV + (CashFlowYearly)/((1+p.DiscountRate)**l) #NPV is a method used to determine the current value of future cash flows generated by the project
        
        NPV = NPV - TotalCost
        ProfIndex = (NPV+TotalCost)/TotalCost # Probability index >1 it's profitable <1 its not
        
        return PayBackTime,DiscountedPayBackTime,NPV,ProfIndex
        
    
    
    
    def geomAnalysis(self,coords):
        #Arrays for pairwise distances and angles
        #Angle 0: x directıon. Angle pi/2: y directıon
        distance=np.zeros((p.numTurbs,p.numTurbs))
        coords = coords[1:,:] # Coordinates of each turbine
        for i in range(0,p.numTurbs):
            distance[i,:]=np.sqrt(np.square(coords[i,0]-coords[:,0])+np.square(coords[i,1]-coords[:,1]))
            distance[i,i]=1e10
        #Rotate angles so that north=0 and convert to degrees
        return distance

    
    
    def minimumSpanningTree(self,distance,numTurbs):
        g=Graph(numTurbs)
        for i in range(0,numTurbs):
            for j in range(i+1,numTurbs):
                g.addEdge(i,j,int(distance[i,j]*1000))
        MSTweight,links=g.boruvkaMST()
        return MSTweight,links      
    
    
    
    
    
    
    def fitness_func(self,ga_instance, solution, solution_idx):
        positionlist = np.reshape(solution, (-1, 2))
        
        # Give a very bad fitness to illegal solutions
        eliminatedSolution = -100_000
        
        # Output the progress of the GA
        percDone = int(ga_instance.generations_completed/p.numGenerationsTest*100)
        print("Fitness Checking Gen :",ga_instance.generations_completed)
        #print(f"{percDone}%",end="\r")
        
        # Cluster the turbines
        indiceslist,u_labels,label = self.clustering_algorithm(positionlist,p.numTurbs,p.maxnodespertree) # calls clustering function
        distance,angle,distance_global,angle_global = self.geom_analysis(positionlist,indiceslist) # calls pre-MST function         
        
        clusters = round(math.ceil(p.numTurbs/p.maximumturbinespertree))
        
        # Check to see if the turbines are too close
        for cluster in range(0,round(math.ceil(p.numTurbs/p.maximumturbinespertree))):
            for row in range(0,len(distance[cluster])):
                for col in range(0,len(distance[cluster])):
                    if distance[cluster][row][col] < p.minTurbGap:
                        # If too close, give a very bad fitness
                        print("TOO CLOSE")
                        return eliminatedSolution  
    
        
        # Calculate the length of cabling required
        MSTweight=self.Minimum_Spanning_Tree(distance,indiceslist)
        MSTWeightSum=0 # Calls MST function
        for a in range(0,math.ceil(p.numTurbs/p.maxnodespertree)): 
            MSTWeightSum += MSTweight[a][0]
        
        # Calculate all of the associated costs
        OnshoreSubstation,ExportDistance = self.Substation_To_Shore_Connection(positionlist)
        FoundationCostTotal = self.foundations(positionlist)
        ExportCableCost = (ExportDistance*p.ExportCableCostperMeter) # Export cable cost
        FixedCost = (p.OffshoreSubstationCostperMW+p.OnshoreSubstationCostperMW)*p.TurbineRating*p.numTurbs
        CableCost= (MSTWeightSum/1000)*p.IACableCostperMeter
        DepthCostAll = self.depthvalues(positionlist)[0]
        TurbineCostTotal= (p.numTurbs)*p.CostperTurbine
        MaintenanceCosts= (p.numTurbs)*p.MaintenanceCostperTurbine
        LandCost = self.LandAreaCost(positionlist,ExportDistance)
        TotalCost= TurbineCostTotal+DepthCostAll+MaintenanceCosts+FixedCost+LandCost+CableCost+ExportCableCost+FoundationCostTotal 
            
        #from CALLGA import predict_class
        print("IMPORTED PREDICT CLASS")
        # Calculate the power production of the wind farm
        wsr=self.windspeedreduction(positionlist[1:,:],p.angles,self.predict_class) # calls wake attenuation function
        powout=self.power(wsr,p.velocities) # total output power
        
        #from CALLGA import wsp
        output=np.tensordot(powout,self.wsp,axes=2) # reduced power output due to wake effects                         
        CostperWatt = TotalCost/(np.sum(output))
        
        
        # Check to see if the budget has been upheld
        if TotalCost > p.budget:
            # If over budget, give a very bad fitness
            print("OVERBUDGET")
            return eliminatedSolution * CostperWatt
        
        print("VALID SOLUTION")
        print(CostperWatt)
        # If solution is valid, return the cost/watt as the fitness
        return -CostperWatt




print("RUNNING AT END OF FUNCTIONS FILE FOR SOME REASON")


