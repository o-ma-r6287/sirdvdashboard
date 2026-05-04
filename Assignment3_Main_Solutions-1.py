import matplotlib.pyplot as plt
import numpy as np
from HW4_Functions import run_sim
#=========================================================================================
# Get user input
#=========================================================================================
#Ask user "Do you want to simulate the SIR, SIRD or SIRDV model?" & assign their input to the variable MODEL_CHOICE
MODEL_CHOICE = input("Do you want to simulate the SIR, SIRD or SIRDV model?").strip().upper()
DAYS = int(input("How many days?"))
#Ask user relevant questions based on the model they chose and store the information in variables.

    # If the user requested a SIR model, ask them 4 questions:
        #"What is the infection rate you want to simulate?" Assign this user input to the variable BETA
        #"What is the recovery rate you want to simulate?" Assign this user input to the variable GAMMA
        #"How many days do you want to simulate the disease outbreak?" Assign this user input to the variable DAYS
        #"Give me a list of 3 numbers: the # of susceptible, # of infected and # of recovered individuals at day 0."
        # Assign the list the user inputted to the variable DAY0_INDV

    # If the user requested a SIRD model, ask the same questions as above but also:
        #"What is the death rate from the disease that you want to simulate?" Assign this user input to the variable MU

    #If the user requested a SIRDV model, ask the same questions as above but also:
        #"What is the vaccination rate you want to simulate?" Assign this user input to the variable VAC_RATE
        # Update the list question to "Give me a list of 4 numbers: the # of susceptible, # of infected, # of recovered
        # individuals and # of vaccinated individuals at day 0." Assign user inputted list to variable DAY0_INDV

# Note: Make sure you convert all the user input numbers to floats!

# The following parameters are common to all models

BETA = float(input("What is the infection rate you want to simulate? "))

GAMMA = float(input("What is the recovery rate you want to simulate? "))

DAY0_INDV = list(map(int, input( "Give me a list of 3 numbers: the # of susceptible, # of infected and # of recovered individuals at day 0: ").split(',')))

MU=0 #just incase. this won't be an issue since our run_sim defaults are 0s.
VAC_RATE=0
if MODEL_CHOICE == "SIRD":
    MU = float(input("What is the death rate from the disease that you want to simulate?"))

elif MODEL_CHOICE == "SIRDV":
    MU = float(input("What is the death rate from the disease that you want to simulate?"))
    VAC_RATE = float(input("What is the vaccination rate you want to simulate? "))
N=DAY0_INDV[0]+DAY0_INDV[1]+DAY0_INDV[2]
#=========================================================================================
# Run default simulation and generate plot. Save the plot as png file called
# DefaultSIR_N1000_100Days_.4,.04_SimPlot.png. The .4, .04 are the infection and recovery rates
#=========================================================================================
print(f"Printing Default SIR Simulation: N= 1000, 100 days, Beta=.4, Gamma=.035")

#Call the run_sim function to run the default SIR Model simulation, save the arrays as Sim_S, Sim_I, Sim_R, Sim_D, Sim_V
Sim_S, Sim_I, Sim_R, Sim_D, Sim_V = run_sim() #Sim_D and Sim_V will be arrays of zeroes bc the SIR Model does not use them

# Create a plot for this default simulation (SIR Model) and make sure it matches the sample output
# (see the "SIR Model.py" starter file listed in the Module 7 Overview for the starter code). If you'd like, you can
# use Seaborn and tweak the plot the way you'd like.

t = np.arange(0, DAYS) #array of time points (0 to number of DAYS the user inputted)
plt.figure(figsize=(10,6))
plt.plot(t, Sim_S, label="Susceptible")
plt.plot(t, Sim_I, label="Infected")
plt.plot(t, Sim_R, label="Recovered")
plt.xlabel("Days \n Simulation Parameters: S=997, I=3, Beta = .4, Gamma = .035. ")
plt.ylabel("Number of People")
plt.title("Default SIR Model (N=1000, 100 Days)")
plt.legend()
plt.grid(True)
plt.savefig("DefaultSIR_N1000_100Days_.4,.035_SimPlot.png")
plt.show() #Make sure to do plt.savefig before you type plt.show() otherwise, the plot will be empty

#=========================================================================================
# Run the simulation the user wants and generate plot. Save the plot as a png file with a name that includes the
# population N (this is S+I+R), the rates (infection, recovery, and if relevant, vaccination and/or death rate), the
# name of the model the user specified followed by _Plot.png

#Example: SIRDSim_N1000_100Days_Rates.4,.05,.1.png
#(if the user requested a SIRD simulation of 100 days for 1000 people with infection,recovery & death rates of .4,.05,.1)
#Hint use an f-string with your variables that stored the user-specified input!
#=========================================================================================

print(f"Printing your Requested Simulation: {MODEL_CHOICE}, N={N} {DAYS} days, Beta={BETA}, Gamma={GAMMA}, Mu={MU}, VAC_RATE={VAC_RATE}")
Sim_S, Sim_I, Sim_R, Sim_D, Sim_V = run_sim(S_0=DAY0_INDV[0], I_0=DAY0_INDV[1], R_0=DAY0_INDV[2], beta=BETA, gamma=GAMMA, mu=MU, vac_rate=VAC_RATE,days=DAYS, model_choice=MODEL_CHOICE)

#Need to use if elif and else conditions to print the correct plot depending on the user's model choice
#Change code below to use f-strings

t = np.arange(0, DAYS) #array of time points (0 to number of DAYS the user inputted)
plt.figure(figsize=(10,6))
plt.plot(t, Sim_S, label="Susceptible")
plt.plot(t, Sim_I, label="Infected")
plt.plot(t, Sim_R, label="Recovered")
plt.plot(t, Sim_D, label="Deceased")
plt.plot(t, Sim_V, label="Vaccinated")
plt.xlabel(f"Days \n Simulation Parameters: S={DAY0_INDV[0]}, I={DAY0_INDV[1]},R={DAY0_INDV[2]}, Beta = .{BETA}, Gamma = {GAMMA}, mu = {MU}, and VAC_RATE={VAC_RATE}")
plt.ylabel("Number of People")
plt.title("User SIRD Model (N=1000, 100 Days)")
plt.legend()
plt.grid(True)
plt.savefig(f"{MODEL_CHOICE}_N{N}_{DAYS}_{BETA},{GAMMA}, {MU}_Simplot.png")
plt.show()

